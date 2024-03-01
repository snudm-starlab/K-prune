#############################################V#########################################
# prune.py
# - implementation of major functions of Kprune
# - including knowledge measurement, knowledge-preserving mask search (KPMS),
#   and knowledge-preserving weight-tuning (KPWT)
#######################################################################################

import torch
import torch.nn.functional as F
from torch.nn import MSELoss
import time
from utils.arch import *


def kprune(model, dataloader, IS_SQUAD, head_dim, lam_pred, lam_rep, mu, T,
              model_flops, f_head, f_neuron, head_masks, neuron_masks,
              constraint, sublayerwise_tuning, logger):
    """
    a function for Kprune which includes knowledge measurement,
    knowledge-preserving mask search (KPMS), snd knowledge-preserving pruning (KPP)

    * Inputs
       - model: the target model to compress
       - dataloader: dataloader for the sample dataset
       - IS_SQUAD: a flag indicating the target task is whether SQuAD or not (GLUE)
       - head_dim: the projected dimension in attention heads (d_h in our paper)
       - lam_pred: a balance coefficient for predictive knowledge
       - lam_rep: a balance coefficient for representational knowledge
       - mu: a balance coefficient for importance scores of attention heads
       - T: the temperature of the softmax function
       - model_flops: number of FLOPs of the uncompressed model
       - f_head: the number of FLOPs for computing an output of an attention head
       - f_neuron: the number of FLOPs for computing an output of a neuron
       - head_masks: initialized masks for attention heads
       - neuron_masks: initialized masks for neurons
       - constraint: the ratio of FLOPs to be reduced 
       - sublayerwise_tuning: whether perform sub-layerwise tuning or not
       - logger: logger for Kprune
    * Outputs:
       - model: a pruned model
       - head_masks: the found pruning masks for attention heads
       - neuron_masks: the found pruning masks for attention heads
    """

    # Initialization
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    intermediate_dim = model.config.intermediate_size
    max_flops = model_flops * (1. - constraint)

    kd_outputs = {}
    kd_labels = []
    for sl in range(2 * num_layers):
        _st = time.time()
        _str = ''
        is_ffn = (sl % 2 == 1)
        _idx = sl // 2
        flops = f_neuron if is_ffn else f_head
        # (1) Knowledge measurement
        #    - Measure the amount of knowledge in the above sub-layers
        head_scores, neuron_scores, features, _inputs, kd_outputs, kd_labels = \
            compute_scores(model, dataloader, IS_SQUAD, sl, head_dim, num_layers,
                           num_heads, intermediate_dim, lam_pred, lam_rep, mu, T, f_head, f_neuron,
                           kd_outputs, kd_labels)

        # (2) Knowledge-preserving mask search (KPMS)
        #    - find the combination of masks that satisfies the FLOPs constraint
        #      and maximize importance scores simultaneously.

        # mask search
        pruning_mask = find_mask(sl, head_scores, neuron_scores, f_head, f_neuron, max_flops)
        if is_ffn:
            _str += f"({_idx:2d}) [FFN] {(~pruning_mask).sum().item()} neurons pruned"
        else:
            _str += f"({_idx:2d}) [MHA] {(~pruning_mask).sum().item()} heads pruned"

        # update mask
        if is_ffn:
            neuron_masks[_idx] = pruning_mask
        else:
            head_masks[_idx] = pruning_mask

        # (3) Knowledge-preserving weight-tuning (KPWT)
        #    - perform pruning selected attention heads (or neurons) in the target sub-layer
        #      and tuning weights to reconstruct knowledge of the PLM
        #    - update the sparsity constraint for the remained sub-layers

        # find the new weights for the pruned layer
        if sublayerwise_tuning:
            try:
                W_new = knowledge_reconstruction(model, pruning_mask, sl, features,
                                                 _inputs, kd_outputs[sl].cuda(), head_dim)
            except RuntimeError:
                # pass
                W_new = None
        else:
            W_new = None

        # pruning
        if is_ffn:
            prune_neurons(model, neuron_masks, _idx, W_new)
        else:
            prune_heads(model, head_masks, _idx, W_new)

        # update the sparsity constraint for the remained sub-layers
        max_flops -= torch.sum(pruning_mask) * flops

        _str += f" ({time.time() - _st:.1f}s)"
        logger.info(_str)

    return model, head_masks, neuron_masks


def compute_scores(model, dataloader, IS_SQUAD, target, head_dim, num_layers, num_heads,
                   intermediate_dim, lam_pred, lam_rep, mu, T, f_head, f_neuron, kd_outputs, kd_labels):
    """
    a function for computing scores of attention heads and neurons.
    this function collects the intermediate outputs and soft-label predictions of the uncompressed
    model when target is zero. 
    this function returns the generated intermediate features and the input of the target sub-layer
    for knowledge reconstruction.

    * Inputs
       - model: target model to compress
       - dataloader: dataloader for the sample dataset
       - IS_SQUAD: flag indicating the target task is whether SQuAD or not (GLUE)
       - target: index of the target sub-layer
       - head_dim: dimension of the attention head outputs (d_h in our paper)
       - num_layers: the number of layers in the model
       - num_heads: the number of attention heads in each sub-layer
       - intermediate_dim: the number of neurons in each sub-layer
       - lam_pred: a balance coefficient for predictive knowledge
       - lam_rep: a balance coefficient for representational knowledge
       - mu: a balance coefficient for attention heads
       - T: the temperature of the softmax function
       - f_head: the number of FLOPs for an attention head 
       - f_neuron: the number of FLOPs for a neuron
       - kd_outputs: the intermediate outputs of sub-layers in the uncompressed model
                    for knowledge reconstruction
       - kd_labels: the soft label prediction of the uncompressed model for measuring predictive
                    knowledge
    * Outputs:
       - head_scores: estimated importance scores for attention heads
       - neuron_scores: estimated importance scores for neurons
       - _features: generated intermediate features of the target sub-layer for knowledge reconstruction 
       - s_inputs: input of the target sub-layer for knowledge reconstruction
       - kd_outputs: the intermediate outputs of sub-layers in the uncompressed model
                    for knowledge reconstruction
       - kd_labels: the soft label prediction of sub-layers for measuring predictive
                    knowledge
    """
    # (1) Initialization
    model.eval()
    # tensors for accumulating the amount of knowledge
    head_label_ks = torch.zeros(num_layers, num_heads).cuda()  # 12x12
    head_rep_ks = torch.zeros(num_layers, num_heads).cuda()  # 12x12
    neuron_label_ks = torch.zeros(num_layers, intermediate_dim).cuda()  # 12 x 3072
    neuron_rep_ks = torch.zeros(num_layers, intermediate_dim).cuda()  # 12 x 3072

    # tensors for compute gradients of masks
    _head_masks = torch.ones(num_layers, num_heads * head_dim).cuda()  # 12 x 768
    _neuron_masks = torch.ones(num_layers, intermediate_dim).cuda()  # 12 x 3072
    _head_masks.requires_grad_(True)
    _neuron_masks.requires_grad_(True)

    # (2) Register hook functions
    handles = []
    _inputs = {}
    sl_features = []
    _outputs = []
    s_inputs = []

    for sl in range(target, 2 * num_layers):
        is_ffn = (sl % 2 == 1)  # sub-layer type
        _idx = sl // 2  # index for layer
        _inputs[sl] = []

        # Find the linear layer to be hooked
        if is_ffn:
            input_proj = get_fc1(model, _idx).dense
            output_proj = get_fc2(model, _idx).dense
            layer_mask = _neuron_masks[_idx]
        else:
            input_proj = get_mha_query_proj(model, _idx)
            output_proj = get_mha_output_proj(model, _idx).dense
            layer_mask = _head_masks[_idx]

        # Register hook functions
        # For sub-layer tuning
        if sl == target:
            handles.append(
                hijack(output_proj, _outputs, _hijack_input=False,
                       _stop_forward=False)
            )
            handles.append(
                hijack(input_proj, s_inputs, _hijack_input=True,
                       _stop_forward=False)
            )
        if target == 0:
            # for kd_outputs at the first iteration
            kd_outputs[sl] = []
            handles.append(
                hijack(input_proj, kd_outputs[sl], _hijack_input=True,
                       _stop_forward=False)
            )

            handles.append(
                update_output(output_proj, kd_outputs[sl])
            )

        # To compute score            
        handles.append(
            hijack(output_proj, _inputs[sl], _hijack_input=True,
                   _stop_forward=False)
        )
        handles.append(
            apply_mask(output_proj, layer_mask)
        )

    # (3) Do forward and measure knowledge
    num_tokens = 0
    num_samples = 0
    _bc = 0
    _index = 0
    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        att_mask = batch['attention_mask'].bool()
        num_tokens += batch['attention_mask'].sum()
        batch_samples = batch['attention_mask'].shape[0]
        num_samples += batch_samples

        outputs = model(**batch)

        # use cross-entropy loss
        if IS_SQUAD:
            kd_labels.append([])  # start_logits
            kd_labels.append([])  # end_logits

        # Compute KL divergence for measuring the amount of predictive knowledge
        if IS_SQUAD:
            # For SQuAD
            if target == 0:
                pred_start = F.softmax(outputs.start_logits / T, dim=1).detach()
                pred_end = F.softmax(outputs.end_logits / T, dim=1).detach()
                kd_labels[0].append(pred_start)
                kd_labels[1].append(pred_end)
            else:
                pred_start = kd_labels[0][_bc]
                pred_end = kd_labels[1][_bc]
                _index += batch_samples
            start_kl_div = F.kl_div(
                input=F.log_softmax(outputs.start_logits / T, dim=1),
                target=pred_start,
                reduction="batchmean"
            ) * (T ** 2)
            end_kl_div = F.kl_div(
                input=F.log_softmax(outputs.end_logits / T, dim=1),
                target=pred_end,
                reduction="batchmean"
            ) * (T ** 2)
            kl_div = (start_kl_div + end_kl_div) / 2
            kl_div.backward()

        else:
            # GLUE
            if model.config.problem_type == 'regression':
                # regression
                loss_fct = MSELoss()
                if target == 0:
                    if model.num_labels == 1:
                        pred = outputs.logits.squeeze().detach()
                    else:
                        pred = outputs.logits.detach()
                    kd_labels.append(pred)
                else:
                    if model.num_labels == 1:
                        pred = kd_labels[_index:_index + batch_samples]
                    else:
                        pred = kd_labels[_index:_index + batch_samples, :]
                    _index += batch_samples
                loss = loss_fct(outputs.logits, pred)
                loss.backward()

            else:
                # single_label_classification
                if target == 0:
                    pred = F.softmax(outputs.logits / T, dim=1).detach()
                    kd_labels.append(pred)
                else:
                    pred = kd_labels[_index:_index + batch_samples, :]
                    _index += batch_samples
                kl_div = F.kl_div(
                    input=F.log_softmax(outputs.logits / T, dim=1),
                    target=pred,
                    reduction="batchmean"
                ) * (T ** 2)
                kl_div.backward()

        # Remove paddings
        _outputs[-1] = remove_paddings(_outputs[-1], att_mask)
        s_inputs[-1] = remove_paddings(s_inputs[-1], att_mask)

        # Measuring the amount of knowledge
        for sl in range(target, 2 * num_layers):
            is_ffn = (sl % 2 == 1)
            _idx = sl // 2  # numbering for layer
            _features = remove_paddings(_inputs[sl][-1], att_mask)

            if target == 0:
                kd_outputs[sl][-1] = remove_paddings(kd_outputs[sl][-1], att_mask).cpu()

            if sl == target:
                sl_features.append(_features)

            if is_ffn:
                # For representational knowledge
                _weight = get_fc2(model, _idx).dense.weight
                neuron_rep_ks[_idx] += \
                    ((_features ** 2).sum(dim=0) *
                     (_weight ** 2).mean(dim=0)).data

                # For predictive knowledge
                layer_grad = _neuron_masks.grad[_idx]
                neuron_label_ks[_idx] += (layer_grad.detach() ** 2) * 0.5

            else:
                # For representational knowledge
                _weight = get_mha_output_proj(model, _idx).dense.weight
                _rep_score = ((_features ** 2).sum(dim=0) *
                              (_weight ** 2).mean(dim=0)) \
                    .view(-1, head_dim).mean(dim=1).data
                head_rep_ks[_idx] += (_rep_score)

                # For predictive knowledge
                layer_grad = _head_masks.grad[_idx]
                _label_score = ((layer_grad.detach() ** 2) * 0.5) \
                    .view(-1, head_dim).mean(dim=1)
                head_label_ks[_idx] += (_label_score)
                pass
            del _inputs[sl][-1]
        _bc += 1
        _neuron_masks.grad = None
        _head_masks.grad = None

    # (4) Finishing
    # Concatenating
    _features = torch.cat(sl_features, dim=0)
    _outputs = torch.cat(_outputs, dim=0)
    s_inputs = torch.cat(s_inputs, dim=0)
    if target == 0:
        if IS_SQUAD:
            pass
        else:
            kd_labels = torch.cat(kd_labels, dim=0)
        for sl in range(2 * num_layers):
            kd_outputs[sl] = torch.cat(kd_outputs[sl], dim=0)

    # Averaging
    head_label_ks = head_label_ks / num_samples
    head_rep_ks = head_rep_ks / num_tokens
    neuron_label_ks = neuron_label_ks / num_samples
    neuron_rep_ks = neuron_rep_ks / num_tokens

    # Compute scores
    head_scores = mu * (head_rep_ks * lam_rep + head_label_ks * lam_pred) / f_head
    neuron_scores = (neuron_rep_ks * lam_rep + neuron_label_ks * lam_pred) / f_neuron

    remove_handles(handles)
    del _inputs

    return head_scores, neuron_scores, _features, s_inputs, kd_outputs, kd_labels


def find_mask(target, head_scores, neuron_scores, f_head, f_neuron, max_flops):
    """
    find a pruning mask that maximizes the sum of importance scores of the pruned model,
    and simultaneously satisfies FLOPs constraint.
    we find the pruning maks considering all remained masks, and only returns the masks
    of the target layer for KPP.

    * Inputs
       - target: an index of the target sub-layer
       - head_scores: estimated importance scores for attention heads
       - neuron_scores: estimated importance scores for neurons
       - f_head: the number of FLOPs for computing an output of an attention head
       - f_neuron: the number of FLOPs for computing an output of a neuron
       - max_flops: maximum number of FLOPs (FLOPs constraint)
    * Outputs:
       - pruning mask: found pruning mask for the target sub-layer (1: survived, 0: pruned)
    """
    # Find the mask considering the scores and FLOPs constraint
    # (1) initialization
    is_ffn = (target % 2 == 1)
    _idx = target // 2
    if is_ffn:
        _head_scores = head_scores[_idx + 1:, :]
        target_scores = neuron_scores[_idx, :]
    else:
        _head_scores = head_scores[_idx:, :]
        target_scores = head_scores[_idx, :]
    _neuron_scores = neuron_scores[_idx:, :]

    # (2) find threshold for the given constraint
    # sorting
    s_tilde = torch.cat((_head_scores.view(-1), _neuron_scores.view(-1)), dim=0).sort().values

    # find theshold
    _thres = s_tilde[0]
    for f in range(_neuron_scores.numel() + _head_scores.numel()):
        _thres = s_tilde[f]
        _head_flops = f_head * (_head_scores > _thres).sum()
        _neuron_flops = f_neuron * (_neuron_scores > _thres).sum()
        if _head_flops + _neuron_flops < max_flops:
            break
    pruning_mask = (target_scores > _thres)  # 1: survived, 0: pruned
    return pruning_mask


def knowledge_reconstruction(model, pruning_mask, target, features,
                             _inputs, kd_output, head_dim):
    """
    reconstruct the knowledge of the uncompressed model by solving least square problem.

    * Inputs
       - model: the target model to prune
       - pruning_mask: pruning mask for the target sub-layer
       - target: an index of the target sub-layer
       - features: generated intermediate features of the target sub-layer for knowledge reconstruction 
       - _inputs: input of the target sub-layer for knowledge reconstruction
       - kd_outputs: the intermediate outputs of sub-layers in the uncompressed model
                    for knowledge reconstruction
       - head_dim: dimension of the attention head outputs (d_h in our paper)

    * Outputs:
       - W_new: the weight that minimize the loss of representational knowledge
    """
    # pruning_mask => 1: survived, 0: pruned
    is_ffn = (target % 2 == 1)
    _idx = target // 2
    if is_ffn:
        output_proj = get_fc2(model, _idx)
    else:
        output_proj = get_mha_output_proj(model, _idx)
        pruning_mask = torch.repeat_interleave(pruning_mask, head_dim)

    _features = features[:, pruning_mask]
    _bias = output_proj.dense.bias

    ATA = _features.T @ _features
    ATB = _features.T @ (kd_output - _inputs - _bias)

    res = torch.linalg.lstsq(ATA, ATB, rcond=0.)
    W_new = res.solution.cuda()
    return W_new
