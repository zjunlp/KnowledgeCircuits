from typing import Optional, List, Union, Literal, Tuple
from functools import partial 

import pandas as pd
import torch 
from torch.nn.functional import kl_div
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

def get_metric(metric_name: str, task: str, tokenizer:Optional[PreTrainedTokenizer]=None, model: Optional[HookedTransformer]=None):
    if metric_name == 'kl_divergence' or metric_name == 'kl':
        return partial(divergence, divergence_type='kl')
    elif metric_name == 'js_divergence' or metric_name == 'js':
        return partial(divergence, divergence_type='js')
    elif metric_name == 'logit_diff' or metric_name == 'prob_diff':
        prob = (metric_name == 'prob_diff')
        if 'greater-than' in task:
            if tokenizer is None:
                if model is None:
                    raise ValueError("Either tokenizer or model must be set for greater-than and prob / logit diff")
                else:
                    tokenizer = model.tokenizer
            logit_diff_fn = get_logit_diff_greater_than(tokenizer)
        elif 'hypernymy' in task:
            logit_diff_fn = logit_diff_hypernymy
        elif task == 'sva':
            if model is None:
                raise ValueError("model must be set for sva and prob / logit diff")
            logit_diff_fn = get_logit_diff_sva(model)
        else:
            logit_diff_fn = logit_diff
        return partial(logit_diff_fn, prob=prob)
    else: 
        raise ValueError(f"got bad metric_name: {metric_name}")

def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def js_div(p: torch.tensor, q: torch.tensor):
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    m = (0.5 * (p + q)).log()
    return 0.5 * (kl_div(m, p.log(), log_target=True, reduction='none').mean(-1) + kl_div(m, q.log(), log_target=True, reduction='none').mean(-1))

def divergence(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, divergence_type: Union[Literal['kl'], Literal['js']]='kl', mean=True, loss=True):
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    probs = torch.softmax(logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    if divergence_type == 'kl':
        results = kl_div(probs.log(), clean_probs.log(), log_target=True, reduction='none').mean(-1)
    elif divergence_type == 'js':
        results = js_div(probs, clean_probs)
    else: 
        raise ValueError(f"Expected divergence_type of 'kl' or 'js', but got '{divergence_type}'")
    return results.mean() if mean else results

def logit_diff(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False):
    clean_logits = get_logit_positions(clean_logits, input_length)
    cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
    good_bad = torch.gather(cleans, -1, labels.to(cleans.device))
    results = good_bad[:, 0] - good_bad[:, 1]

    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean: 
        results = results.mean()
    return results
    
def direct_logit(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False):
    clean_logits = get_logit_positions(clean_logits, input_length)
    cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
    good_bad = torch.gather(cleans, -1, labels.to(cleans.device))
    results = good_bad[:, 0]

    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean: 
        results = results.mean()
    return results

def logit_diff_hypernymy(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: List[torch.Tensor], mean=True, prob=False, loss=False):
    clean_logits = get_logit_positions(clean_logits, input_length)
    cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits

    results = []
    for i, (ls,corrupted_ls) in enumerate(labels):
        r = cleans[i][ls.to(cleans.device)].sum() - cleans[i][corrupted_ls.to(cleans.device)].sum()
        results.append(r)
    results = torch.stack(results)
    
    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean: 
        results = results.mean()
    return results

def get_year_indices(tokenizer: PreTrainedTokenizer):
    return torch.tensor([tokenizer(f'{year:02d}').input_ids[0] for year in range(100)])


def get_logit_diff_greater_than(tokenizer: PreTrainedTokenizer):
    year_indices = get_year_indices(tokenizer) 
    def logit_diff_greater_than(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False):
        # Prob diff (negative, since it's a loss)
        clean_logits = get_logit_positions(clean_logits, input_length)
        cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
        cleans = cleans[:, year_indices]

        results = []
        if prob:
            for prob, year in zip(cleans, labels):
                results.append(prob[year + 1 :].sum() - prob[: year + 1].sum())
        else:
            for logit, year in zip(cleans, labels):
                results.append(logit[year + 1 :].mean() - logit[: year + 1].mean())

        results = torch.stack(results)
        if loss:
            results = -results
        if mean: 
            results = results.mean()
        return results
    return logit_diff_greater_than

def get_singular_and_plural(model, strict=False) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = model.tokenizer
    tokenizer_length = model.cfg.d_vocab_out

    df: pd.DataFrame = pd.read_csv('data/sva/combined_verb_list.csv')
    singular = df['sing'].to_list()
    plural = df['plur'].to_list()
    singular_set = set(singular)
    plural_set = set(plural)
    verb_set = singular_set | plural_set
    assert len(singular_set & plural_set) == 0, f"{singular_set & plural_set}"
    singular_indices, plural_indices = [], []

    for i in range(tokenizer_length):
        token = tokenizer._convert_id_to_token(i)
        if token is not None:
            if token[0] == 'Ġ':
                token = token[1:]
                if token in verb_set:    
                    if token in singular_set:
                        singular_indices.append(i)
                    else:  # token in plural_set:
                        idx = plural.index(token)
                        third_person_present = singular[idx]
                        third_person_present_tokenized = tokenizer(f' {third_person_present}', add_special_tokens=False)['input_ids']
                        if len(third_person_present_tokenized) == 1 and third_person_present_tokenized[0] != tokenizer.unk_token_id:
                            plural_indices.append(i)
                        elif not strict:
                            plural_indices.append(i)
               
    return torch.tensor(singular_indices, device=model.cfg.device), torch.tensor(plural_indices, device=model.cfg.device)

def get_logit_diff_sva(model, strict=True) -> torch.Tensor:
    singular_indices, plural_indices = get_singular_and_plural(model, strict=strict)
    def sva_logit_diff(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False):
        clean_logits = get_logit_positions(clean_logits, input_length)
        cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
        
        if prob:
            singular = cleans[:, singular_indices].sum(-1)
            plural = cleans[:, plural_indices].sum(-1)
        else:
            singular = cleans[:, singular_indices].mean(-1)
            plural = cleans[:, plural_indices].mean(-1)

        results = torch.where(labels.to(cleans.device) == 0, singular - plural, plural - singular)
        if loss: 
            results = -results
        if mean:
            results = results.mean()
        return results
    return sva_logit_diff