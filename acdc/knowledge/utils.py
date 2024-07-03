import dataclasses
from functools import partial
from acdc.docstring.utils import AllDataThings
import wandb
import os
import logging
from collections import defaultdict
import pickle
import torch
import huggingface_hub
import datetime
from typing import Dict, Callable
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)
import warnings
import networkx as nx
from acdc.knowledge.knowledge_dataset import get_and_filter_dataset
from acdc.acdc_utils import (
    MatchNLLMetric,
    make_nd_dict,
    shuffle_tensor,
)
import transformers
import numpy as np
from acdc.TLACDCEdge import (
    TorchIndex,
    Edge, 
    EdgeType,
)  # these introduce several important classes !!!
from transformer_lens import HookedTransformer
from acdc.acdc_utils import kl_divergence, negative_log_probs
logger = logging.getLogger(__name__)


GPT_J_NAME_SHORT = "gptj"  # A useful alias for the CLI.
GPT_J_NAME = "EleutherAI/gpt-j-6B"

GPT_NEO_X_NAME_SHORT = "neox"
GPT_NEO_X_NAME = "EleutherAI/gpt-neox-20b"

LLAMA_13B_NAME = "llama-13b"
LLAMA_30B_NAME = "llama-30b"
LLAMA_NAME_SHORT = "llama"

# DOWNLOADABLE_MODELS = frozenset({GPT_J_NAME, GPT_NEO_X_NAME, "gpt2-xl"})

def load_model(
    name: str, device = torch.device('cpu'), fp16: Optional[bool] = None
):
    """Load the model given its string name.

    Args:
        name: Name of the model or path to it.
        device: If set, send model to this device. Defaults to CPU.
        fp16: Whether to use half precision. If not set, depends on model.
    Returns:
        ModelAndTokenizer: Loaded model and its tokenizer.
    """
    if name == GPT_J_NAME_SHORT:
        name = GPT_J_NAME
    elif name == GPT_NEO_X_NAME_SHORT:
        name = GPT_NEO_X_NAME
    elif name == LLAMA_NAME_SHORT:
        name = LLAMA_13B_NAME

    # I usually save randomly initialized variants under the short name of the
    # corresponding real model (e.g. gptj_random, neox_random), so check here
    # if we are dealing with *any* variant of the big model.
    is_gpt_j_variant = name == GPT_J_NAME or GPT_J_NAME_SHORT in name
    is_neo_x_variant = name == GPT_NEO_X_NAME or GPT_NEO_X_NAME_SHORT in name
    is_llama_variant = (
        name in {LLAMA_13B_NAME, LLAMA_30B_NAME} or LLAMA_NAME_SHORT in name
    )

    if fp16 is None:
        fp16 = is_gpt_j_variant or is_neo_x_variant or is_llama_variant

    torch_dtype = torch.float16 if fp16 else None

    kwargs: dict = dict(torch_dtype=torch_dtype)
    if is_gpt_j_variant:
        kwargs["low_cpu_mem_usage"] = True
        if fp16:
            kwargs["revision"] = "float16"

    logger.info(f"loading {name} (fp16={fp16})")

    model = transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.to(torch_dtype)
    model.to(device)
    model.eval()
    print(is_llama_variant)
    if is_llama_variant:
        tokenizer = transformers.LlamaTokenizerFast.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token = "</s>"
        tokenizer.pad_token_id = tokenizer.eos_token_id = 2
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def get_model(name, hf_model, tokenizer, device="cuda",local_path=None) -> HookedTransformer:
    if 'Llama' in name or 'llama' in name:
        tl_model =HookedTransformer.from_pretrained(name, hf_model=hf_model, device="cpu", fold_ln=False, dtype=torch.float16,
                                                    local_path=local_path, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)
    else:
        tl_model = HookedTransformer.from_pretrained(name, hf_model=hf_model, tokenizer=tokenizer,local_path=local_path)
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True) 
    #改了这个地方后面绘图应该会报错
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    # logger.info(
    #     f"dtype: {tl_model.type}, device: {tl_model.device}, memory: {tl_model.get_memory_footprint()}"
    # )
    return tl_model


def get_all_knowledge_things(num_examples, device, model="gpt2", model_path="",knowledge_type="", data_path="", relation_name="",
                             reverse = False, data_seed=42, metric_name="match_nll", return_one_element=True) -> AllDataThings:
    hf_model, tokenizer = load_model(model_path)
    tl_model = get_model(name=model, hf_model=hf_model, tokenizer=tokenizer,device=device,local_path=model_path)
    knowledge_data, knowledge_label = get_and_filter_dataset(
        model=tl_model,
        tokenizer=tokenizer,
        knowledge_type=knowledge_type,
        relation_name=relation_name,
        data_path=data_path,
        reverse=reverse,
    )
    default_data = knowledge_data.to(device)
    labels = knowledge_label.to(device)
    validation_data = default_data[:num_examples, :]
    validation_patch_data = shuffle_tensor(validation_data, seed=data_seed).contiguous()
    validation_labels = labels[:num_examples]
    
    test_data = default_data[num_examples:num_examples*2, :]
    test_patch_data = shuffle_tensor(test_data, seed=data_seed).contiguous()
    test_labels = labels[num_examples:num_examples*2]

    with torch.no_grad():
        base_val_logprobs = F.log_softmax(tl_model(validation_data), dim=-1).detach()
        base_test_logprobs = F.log_softmax(tl_model(test_data), dim=-1).detach()

    if metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_val_logprobs,
            return_one_element=return_one_element,
        )
    elif metric_name == "nll":
        validation_metric = partial(
            negative_log_probs,
            labels=validation_labels,
            last_seq_element_only=False,
            shift=True,
        )
    elif metric_name == "match_nll":
        validation_metric = MatchNLLMetric(
            labels=validation_labels, base_model_logprobs=base_val_logprobs,last_seq_element_only=False,shift=True,
        )
    else:
        raise ValueError(f"Unknown metric {metric_name}")

    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
        ),
        "nll": partial(
            negative_log_probs,
            labels=test_labels,
            last_seq_element_only=False,
            shift=True,
        ),
        "match_nll": MatchNLLMetric(
            labels=test_labels, base_model_logprobs=base_test_logprobs,last_seq_element_only=False, shift=True,
        ),
    }
    return AllDataThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_patch_data=validation_patch_data,
        validation_mask=None,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_patch_data=test_patch_data,
        test_mask=None
    )


def one_item_per_batch(toks_int_values, toks_int_values_other, mask_rep, base_model_logprobs, kl_take_mean=True):
    """Returns each instance of induction as its own batch idx"""

    end_positions = []
    batch_size, seq_len = toks_int_values.shape
    new_tensors = []

    toks_int_values_other_batch_list = []
    new_base_model_logprobs_list = []

    for i in range(batch_size):
        for j in range(seq_len - 1): # -1 because we don't know what follows the last token so can't calculate losses
            if mask_rep[i, j]:
                end_positions.append(j)
                new_tensors.append(toks_int_values[i].cpu().clone())
                toks_int_values_other_batch_list.append(toks_int_values_other[i].cpu().clone())
                new_base_model_logprobs_list.append(base_model_logprobs[i].cpu().clone())

    toks_int_values_other_batch = torch.stack(toks_int_values_other_batch_list).to(toks_int_values.device).clone()
    return_tensor = torch.stack(new_tensors).to(toks_int_values.device).clone()
    end_positions_tensor = torch.tensor(end_positions).long()

    new_base_model_logprobs = torch.stack(new_base_model_logprobs_list)[torch.arange(len(end_positions_tensor)), end_positions_tensor].to(toks_int_values.device).clone()
    metric = partial(
        kl_divergence, 
        base_model_logprobs=new_base_model_logprobs, 
        end_positions=end_positions_tensor, 
        mask_repeat_candidates=None, # !!! 
        last_seq_element_only=False, 
        return_one_element=False
    )
    return return_tensor, toks_int_values_other_batch, end_positions_tensor, metric

def get_hit10(logits,knowledge_data):
    # test_logprobs = F.log_softmax(logits, dim=-1)[:,-2,:]
    labels = knowledge_data[:,-1]
    # original_loss = F.nll_loss(test_logprobs, labels, reduction="none")
    hit_at_10 = 0
    labels = labels.squeeze().detach().cpu().numpy().tolist()
    logits = logits[:, -2, :].detach().cpu().numpy()
    for logit, label in zip(logits, labels):
        rank = np.argsort(logit)[::-1].tolist().index(label) + 1
        if rank <= 10:
            hit_at_10 += 1
    hit_at_10_accuracy = hit_at_10 / len(labels)
    print('hit10', hit_at_10_accuracy)
    return hit_at_10_accuracy