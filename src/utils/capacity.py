#######################################################################################
# utils/capacity.py
# - codes for computing the number of FLOPs and parameters of the model
#######################################################################################

# FLOPs
# Ignore the FLOPs for activation, normalization, bias addition
def flops_per_head(seq_len, head_dim, embed_dim):
    # return FLOPs for computing an output of an attention head
    qkv_proj = 3 * 2 * embed_dim * head_dim * seq_len
    att = 2 * 2 * seq_len * seq_len * head_dim
    output_proj = 2 * head_dim * seq_len * embed_dim
    return qkv_proj + att + output_proj


def flops_per_neuron(seq_len, embed_dim):
    # return FLOPs for computing an output of a neuron
    return 2 * 2 * seq_len * embed_dim


def get_flops(num_head_list, num_neuron_list, seq_len, head_dim, embed_dim):
    """
    compute the number of FLOPs of the model
    * Inputs
        - num_head_list: a list of numbers of attention heads in each MHA sub-layer
        - num_neuron_list: a list of numbers of neuron in each FFN sub-layer
        - seq_len: length of the sequence
        - head_dim: a projected dimension in attention heads
        - embed_dim: an embedding dimension
    * Outputs
        - number of FLOPs of the model

    """
    FLOPs = 0.
    for nh, nn in zip(num_head_list, num_neuron_list):
        FLOPs += (nh * flops_per_head(seq_len, head_dim, embed_dim)
                  + nn * flops_per_neuron(seq_len, embed_dim))
    return FLOPs


# Number of parameters
# Do not count embedding parameters
def get_params(model, include_embedding=False):
    _num = 0
    keys = ["embedding", "layer_transformation", "classifier", "pooler"]
    for _n, _p in model.named_parameters():
        if not include_embedding:
            if not any(_k in _n for _k in keys):
                _num += _p.numel()
    return _num
