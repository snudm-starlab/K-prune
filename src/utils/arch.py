#############################################V#######################################
# utils/arch.py
# - codes for utility functions related to the architecture of the model
# - this file includes functions for accessing modules, registering hook 
#   functions, pruning. 
######################################################################################

import torch
import torch.nn as nn
from types import MethodType


###########################################
###### functions for accessing layers #####
###########################################

def get_basemodel(_model):
    _base_model = _model.base_model_prefix
    return getattr(_model, _base_model)


def get_encoder(_model):
    return get_basemodel(_model).encoder


def get_layers(_model):
    return get_encoder(_model).layer


def get_layer(_model, _idx):
    return get_layers(_model)[_idx]


def get_mha_attention(_model, _idx):
    return get_layer(_model, _idx).attention


def get_mha_self_attention(_model, _idx):
    return get_layer(_model, _idx).attention.self


def get_mha_query_proj(_model, _idx):
    return get_layer(_model, _idx).attention.self.query


def get_mha_key_proj(_model, _idx):
    return get_layer(_model, _idx).attention.self.key


def get_mha_value_proj(_model, _idx):
    return get_layer(_model, _idx).attention.self.value


def get_mha_output_proj(_model, _idx):
    return get_layer(_model, _idx).attention.output


def get_fc1(_model, _idx):
    return get_layer(_model, _idx).intermediate


def get_fc2(_model, _idx):
    return get_layer(_model, _idx).output


def get_ln(_model, _idx, is_ffn):
    # the layer normalization 
    _base_model = getattr(_model, _model.base_model_prefix)
    _layer = _base_model.encoder.layer[_idx]
    if is_ffn:
        _output = _layer.output
    else:
        _output = _layer.attention.output
    return _output.LayerNorm


def get_pooler(_model):
    return get_basemodel(_model).pooler


def get_classifier(_model):
    return _model.classifier


###########################################
############# Hook functions ##############
###########################################

class StopFowardException(Exception):
    pass


def hijack(module, _list, _hijack_input, _stop_forward=False):
    # if _stop_forward=True, then it raise error after forwarding the module
    if _hijack_input:
        def input_hook(_, inputs, __):
            _list.append(inputs[0].clone().data)
            if _stop_forward:
                raise StopFowardException

        handle = module.register_forward_hook(input_hook)
    else:
        def output_hook(_, __, outputs):
            _list.append(outputs.clone().data)
            if _stop_forward:
                raise StopFowardException

        handle = module.register_forward_hook(output_hook)
    return handle


def update_output(module, _list):
    # hook function for reducing memory 
    # do not generate additional list and update the existing list
    def update_output_hook(_, __, outputs):
        _list[-1] += outputs.clone().data

    handle = module.register_forward_hook(update_output_hook)
    return handle


def apply_mask(module, _mask):
    # applying masks to the input to compute gradients
    def masking(_, i):
        return _mask * i[0]

    handle = module.register_forward_pre_hook(masking)
    return handle


def remove_handles(handles):
    # remove handles after collecting features
    for handle in handles:
        handle.remove()


###########################################
######## functions for pruning ############
###########################################

def prune_linear_layer(layer, _mask, dim):
    # Input: linear layer, _mask (1 (surv), 0(to prune), -i(pruned)), dim (0: output, 1: input)
    _device = layer.weight.device
    curr_mask = _mask[_mask > (-1e-5)]  # remove already pruned units
    surv_mask = (curr_mask > 1e-5)  # Masking surv=1, to prune 0
    if dim == 1:
        W = layer.weight[:, surv_mask].clone().detach()
    else:
        W = layer.weight[surv_mask, :].clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[surv_mask].clone().detach()

    new_size = list(layer.weight.size())
    new_size[dim] = torch.sum(surv_mask).item()
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(_device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def pruned_BertAttention_forward(self, hidden_states,
                                 attention_mask=None, head_mask=None,
                                 encoder_hidden_states=None, encoder_attention_mask=None,
                                 past_key_value=None, output_attentions=False
                                 ):
    # A modified forward function of a MHA sub-layer for the case that
    # there are no attention heads in the MHA sub-layer
    attention_output = self.output.LayerNorm(hidden_states)
    outputs = (attention_output, None) if output_attentions else (attention_output,)
    return outputs


def pruned_feed_forward_chunk(self, attention_output):
    # A modified forward function in a FFN sub-layer for the case that
    # there are no neurons in the FFN sub-layer
    return self.output.LayerNorm(attention_output)


def prune_sublayer(_model, _idx, is_ffn):
    # We do not remove layer normalization to perserve the original output
    _layer = get_layer(_model, _idx)
    if is_ffn:
        # For FFN sub-layers that do not have neurons
        _layer.intermediate = None
        _layer.output.dense = None
        _layer.output.dropout = None
        _layer.feed_forward_chunk = MethodType(pruned_feed_forward_chunk, _layer)
    else:
        # For MHA sub-layers that do not have attention heads
        _attn = _layer.attention
        _attn.self = None
        _attn.output.dense = None
        _attn.output.dropout = None
        _attn.forward = MethodType(pruned_BertAttention_forward, _attn)


def convert_mask_to_dict(_mask):
    # Mask shape: [layers, num_units]
    # return dict: {layer_idx: [pruning units]}
    num_layers, num_units = _mask.shape
    _d = {}
    for l in range(num_layers):
        _d[l] = torch.arange(num_units)[_mask[l] < 1e-5].tolist()
    return _d


def update_output_proj(_module, new_weight):
    # update the weight of the output projection
    # Note: do not update the bias
    _module.dense.weight.requires_grad = False
    _module.dense.weight.copy_(new_weight.contiguous())
    _module.dense.weight.requires_grad = True


def get_pruning_status(head_masks, neuron_masks):
    # return the number of remained heads and neurons in each sub-layer
    survived_heads = (head_masks > 1e-5).sum(dim=1).tolist()
    survived_neurons = (neuron_masks > 1e-5).sum(dim=1).tolist()
    return survived_heads, survived_neurons


@torch.no_grad()
def prune_heads(_model, head_mask, _idx, new_weight):
    # prune attention heads in an MHA sub-layer and update the weight matrix
    # _idx: an index of the target sub-layer
    # new_weight: tuned weight after knowledge reconstruction
    if (head_mask[_idx] > 1e-3).sum() < 1:
        # remove the MHA sublayer
        prune_sublayer(_model, _idx, is_ffn=False)
    else:
        # paritally pruning
        _attn = get_mha_attention(_model, _idx)
        _self = _attn.self
        layer_mask = torch.repeat_interleave(head_mask[_idx], _self.attention_head_size)

        _self.query = prune_linear_layer(_self.query, layer_mask, dim=0)
        _self.key = prune_linear_layer(_self.key, layer_mask, dim=0)
        _self.value = prune_linear_layer(_self.value, layer_mask, dim=0)

        out_proj = get_mha_output_proj(_model, _idx)
        out_proj.dense = prune_linear_layer(out_proj.dense, layer_mask, dim=1)

        _self.num_attention_heads = torch.sum(head_mask[_idx] > 1e-5)
        _self.all_head_size = _self.attention_head_size * _self.num_attention_heads

        _heads = convert_mask_to_dict(head_mask)
        _attn.pruned_heads = _attn.pruned_heads.union(_heads)

        # upate weights
        if new_weight is not None:
            update_output_proj(out_proj, new_weight.T.contiguous())


@torch.no_grad()
def prune_neurons(_model, neuron_mask, _idx, new_weight):
    # prune neurons in a FFN sub-layer and update the weight matrix
    # _idx: an index of the target sub-layer
    # new_weight: tuned weight after knowledge reconstruction

    if (neuron_mask[_idx] == 0.).sum() < 1e-5:
        # do not pruning
        return
    elif (neuron_mask[_idx] > 1e-3).sum() < 1.:
        # remove the FFN sublayer
        prune_sublayer(_model, _idx, is_ffn=True)
    else:
        # paritally pruning
        _fc1 = get_fc1(_model, _idx)
        _fc2 = get_fc2(_model, _idx)
        layer_mask = neuron_mask[_idx]
        _fc1.dense = prune_linear_layer(_fc1.dense, layer_mask, dim=0)
        _fc2.dense = prune_linear_layer(_fc2.dense, layer_mask, dim=1)

        # upate weights
        if new_weight is not None:
            update_output_proj(_fc2, new_weight.T.contiguous())


@torch.no_grad()
def remove_grads(_model):
    # remove gradients of parameters in the model
    for _p in _model.parameters():
        if _p.requires_grad:
            _p.grad = None


def print_params(_model, ):
    # print names and shapes of parameters in the model
    for _n, _p in _model.named_parameters():
        print(_n, _p.shape)


def remove_paddings(_value, att_mask):
    # remove paddings
    return _value[att_mask, :]
