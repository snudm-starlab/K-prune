###############################################################
# main.py
# - codes for running K-prune
# - parsing arguments, loading data and model, and save results
###############################################################

import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from dataset.glue import glue_dataset, max_seq_length, avg_seq_length
from dataset.squad import squad_dataset
from evaluate.nlp import test_accuracy
from prune import kprune
from utils.capacity import *
from utils.arch import *

# Remove logging from transformers
_data_logger = logging.getLogger("datasets.builder")
_data_logger.setLevel(logging.ERROR)
_data_logger = logging.getLogger("datasets.arrow_dataset")
_data_logger.setLevel(logging.ERROR)

# Generate a new logger
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# General
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--task_name", type=str, required=True, choices=[
    "mnli",
    "qqp",
    "qnli",
    "sst2",
    "stsb",
    "mrpc",
    "squad",
    "squad_v2",
])
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--exp_name", type=str, default='test')
parser.add_argument("--gpu", type=int, default=0)

parser.add_argument("--constraint", type=float, required=True,
                    help="FLOPs constraint, the percetange of FLOPs to remove",
                    )
parser.add_argument("--seed", type=int, default=0)
# For K-prune
parser.add_argument("--sublayerwise_tuning", action='store_true', default=False,
                    help="whether use sub-layerwise tuning or not")
parser.add_argument("--num_tokens", type=int, default=100000,
                    help='the number of tokens to use for K-prune')
parser.add_argument("--lam_pred", type=float, default=1.0,
                    help='a balance coefficient for predictive knowledge')
parser.add_argument("--lam_rep", type=float, default=1e-5,
                    help='a balance coefficient for representational knowledge')
parser.add_argument("--mu", type=float, default=64.,
                    help='a balance coefficient for scores of attention heads')
parser.add_argument("--T", type=float, default=2.,
                    help='a temperature for softmax functions')


def main():
    args = parser.parse_args()
    IS_SQUAD = "squad" in args.task_name
    IS_LARGE = "large" in args.model_name
    seq_len = 170 if IS_SQUAD else avg_seq_length(args.task_name)

    # Create the output directory
    OUTPUT_DIR = os.path.join(
        "outputs",
        args.model_name,
        args.exp_name,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initiate the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(OUTPUT_DIR, "log.txt")),
        ],
    )
    logger.info(args)

    # Set a GPU and the experiment seed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    logger.info(f"Seed number: {args.seed}")

    # Load the fine-tuned model and the corresponding tokenizer
    config = AutoConfig.from_pretrained(args.ckpt_dir)
    if IS_SQUAD:
        model_generator = AutoModelForQuestionAnswering
    else:
        model_generator = AutoModelForSequenceClassification
    model = model_generator.from_pretrained(args.ckpt_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        use_auth_token=None,
    )

    # Load the training dataset
    if IS_SQUAD:
        training_dataset = squad_dataset(
            args.task_name,
            tokenizer,
            training=True,
            max_seq_len=384,
            pad_to_max=False,
        )
    else:
        training_dataset = glue_dataset(
            args.task_name,
            tokenizer,
            training=True,
            max_seq_len=max_seq_length(args.task_name),
            pad_to_max=False,
        )

    batch_size = int((12 if IS_SQUAD else 32) * (0.5 if IS_LARGE else 1))

    # Create the sample dataloder 
    collate_fn = DataCollatorWithPadding(tokenizer)
    num_samples = min(int(args.num_tokens / seq_len), len(training_dataset))
    logger.info(f"Num sampels: {num_samples}")
    sample_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), num_samples).tolist(),
    )

    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    # Prepare the model
    model = model.cuda()
    model.eval()

    for _p in model.parameters():
        _p.requires_grad = False

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    embed_dim = model.config.hidden_size

    # Initialize masks
    head_masks = torch.ones(model.config.num_hidden_layers,
                            model.config.num_attention_heads).cuda()
    neuron_masks = torch.ones(model.config.num_hidden_layers,
                              model.config.intermediate_size).cuda()

    # Get flops and params of the model before pruning
    model_flops = get_flops(
        num_head_list=(head_masks > 0).sum(dim=1).tolist(),
        num_neuron_list=(neuron_masks > 0).sum(dim=1).tolist(),
        seq_len=seq_len,
        head_dim=head_dim,
        embed_dim=embed_dim,
    ) / 1e9
    f_head = flops_per_head(seq_len, head_dim, embed_dim) / 1e9
    f_neuron = flops_per_neuron(seq_len, embed_dim) / 1e9

    st = time.time()

    # Perform K-prune
    # For each sub-layer do follows
    # (1) Knowledge measurement
    #    - Measure the amount of knowledge in the target and above sub-layers
    # (2) Knowledge-preserving mask search (KPMS)
    #    - find the combination of masks that satisfies the FLOPs constraint
    #      and maximize importance scores simultaneously.
    # (3) Knowledge-preserving weight-tuning (KPWT)
    #    - perform pruning selected attention heads (or neurons) in the target sub-layer
    #      and tuning weights to reconstruct knowledge of the PLM
    #    - update the sparsity constraint for the remained sub-layers

    model, head_masks, neuron_masks = kprune(model, sample_dataloader, IS_SQUAD, head_dim,
                                                args.lam_pred, args.lam_rep, args.mu, args.T,
                                                model_flops, f_head, f_neuron, head_masks, neuron_masks,
                                                args.constraint, args.sublayerwise_tuning, logger)

    pruning_time = time.time() - st
    logger.info(f"Time for K-prune: {pruning_time:.3f}")
    # K-Pruning done

    # Get statistics of the pruned model
    survived_heads, survived_neurons = get_pruning_status(head_masks, neuron_masks)
    pruned_model_flops = get_flops(
        num_head_list=survived_heads,
        num_neuron_list=survived_neurons,
        seq_len=seq_len,
        head_dim=head_dim,
        embed_dim=embed_dim,
    ) / 1e9
    flops_pruned_ratio = (1. - pruned_model_flops / model_flops) * 100
    test_acc = test_accuracy(model, tokenizer, args.task_name)

    # Save the model and masks
    head_mask_path = os.path.join(OUTPUT_DIR, 'head_masks.pkl')
    neuron_mask_path = os.path.join(OUTPUT_DIR, 'neuron_masks.pkl')
    ckpt_path = os.path.join(OUTPUT_DIR, 'model.ckpt')
    torch.save(model.cpu(), ckpt_path)
    torch.save(head_masks.cpu(), head_mask_path)
    torch.save(neuron_masks.cpu(), neuron_mask_path)

    logger.info(f"* FLOPs: {model_flops:.2f} -> {pruned_model_flops:.2f}" +
                f" ({flops_pruned_ratio:.4f}%)| acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
