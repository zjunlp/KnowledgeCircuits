
import io
import logging
from logging import warning
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union, List, Literal, Sequence
from site import PREFIXES
import warnings
import torch
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import random
import re
import json
import matplotlib.pyplot as plt
import copy
from dataclasses_json import DataClassJsonMixin

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# PathLike = str | Path
logger = logging.getLogger(__name__)

RelationFnType = Literal["ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_ONE", "MANY_TO_MANY"]


@dataclass(frozen=True)
class RelationSample(DataClassJsonMixin):
    """A single (subject, object) pair in a relation."""

    subject: str
    object: str

    def __str__(self) -> str:
        return f"{self.subject} -> {self.object}"


@dataclass(frozen=True)
class RelationProperties(DataClassJsonMixin):
    """Some metadata about a relation."""

    relation_type: str
    domain_name: str
    range_name: str
    symmetric: bool
    fn_type: str
    disambiguating: bool


@dataclass(frozen=True)
class Relation(DataClassJsonMixin):
    """An abstract mapping between subjects and objects.

    Attributes:
        name: The name of the relation, used as an ID.
        prompt_templates: Prompts representing the relation, where the subject is
            represented by {}.
        samples: A list of (subject, object) pairs satisfying the relation.
        properties: Relation metadata.
        _domain: Explicit list of all possible subjects. Accessed via the @property
            `domain`, which guesses the domain from the samples if not provided.
        _range: Equivalent to `_domain`, but for objects.
    """

    name: str
    prompt_templates: list[str]
    prompt_templates_zs: list[str]
    samples: list[RelationSample]
    properties: RelationProperties

    _domain: list[str] | None = None
    _range: list[str] | None = None

    @property
    def domain(self) -> set[str]:
        if self._domain is not None:
            return set(self._domain)
        return {sample.subject for sample in self.samples}

    @property
    def range(self) -> set[str]:
        if self._range is not None:
            return set(self._range)
        return {sample.object for sample in self.samples}

    def without(self, sample: RelationSample) -> "Relation":
        """Return a copy of this relation without a given sample."""
        return self.set(samples=[s for s in self.samples if s != sample])

    def split(
        self, train_size: int, test_size: int | None = None
    ) -> tuple["Relation", "Relation"]:
        """Break into a train/test split."""
        if train_size > len(self.samples):
            raise ValueError(f"size must be <= {len(self.samples)}, got: {train_size}")
        if test_size is None:
            test_size = len(self.samples) - train_size

        # Shuffle once up front, because we're sometimes sorted, and if the relation
        # is 1:1, we'll always pick the same samples!
        samples = self.samples.copy()
        random.shuffle(samples)

        samples_by_object = defaultdict(list)
        for sample in samples:
            samples_by_object[sample.object].append(sample)

        for samples in samples_by_object.values():
            random.shuffle(samples)

        # List to store the result
        max_coverage_samples = []

        # As long as there are samples left
        while samples_by_object:
            # For each object
            for object in list(samples_by_object.keys()):
                # Add one sample to the result and remove it from the object's list
                max_coverage_samples.append(samples_by_object[object].pop(0))

                # If there are no more samples for this object, remove it from the dict
                if len(samples_by_object[object]) == 0:
                    del samples_by_object[object]

        train_samples = max_coverage_samples[:train_size]
        test_samples = max_coverage_samples[train_size : train_size + test_size]

        return (
            Relation(
                name=self.name,
                prompt_templates=self.prompt_templates,
                prompt_templates_zs=self.prompt_templates_zs,
                properties=self.properties,
                samples=train_samples,
                _domain=list(self.domain),
                _range=list(self.range),
            ),
            Relation(
                name=self.name,
                prompt_templates=self.prompt_templates,
                prompt_templates_zs=self.prompt_templates_zs,
                properties=self.properties,
                samples=test_samples,
                _domain=list(self.domain),
                _range=list(self.range),
            ),
        )

    def set(
        self,
        name: str | None = None,
        prompt_templates: Sequence[str] | None = None,
        prompt_templates_zs: Sequence[str] | None = None,
        properties: RelationProperties | None = None,
        samples: Sequence[RelationSample] | None = None,
        domain: Sequence[str] | None = None,
        range: Sequence[str] | None = None,
    ) -> "Relation":
        """Return a copy of this relation with any specified fields overwritten."""
        return Relation(
            name=name if name is not None else self.name,
            prompt_templates=list(prompt_templates)
            if prompt_templates is not None
            else self.prompt_templates,
            prompt_templates_zs=list(prompt_templates_zs)
            if prompt_templates_zs is not None
            else self.prompt_templates_zs,
            properties=properties if properties is not None else self.properties,
            samples=list(samples) if samples is not None else self.samples,
            _domain=list(domain) if domain is not None else self._domain,
            _range=list(range) if range is not None else self._range,
        )


class RelationDataset(torch.utils.data.Dataset[Relation]):
    """A torch dataset of relations."""

    def __init__(self, relations: list[Relation]):
        self.relations = relations

    def __len__(self) -> int:
        return len(self.relations)

    def __getitem__(self, index: int) -> Relation:
        return self.relations[index]

    def filter(
        self,
        relation_names: Sequence[str] | None = None,
        **properties: bool | Sequence[str],
    ) -> "RelationDataset":
        relations = list(self.relations)
        if relation_names is not None:
            logger.debug(f"filtering to only relations: {relation_names}")
            relations = [r for r in relations if r.name in set(relation_names)]

        for key, value in properties.items():
            if value is not None:
                if isinstance(value, bool):
                    logger.debug(f"filtering by property {key}={value}")
                    matches = lambda x: x == value
                else:
                    logger.debug(f"filtering by property {key} in {value}")
                    value_set = set(value)
                    matches = lambda x: (x in value_set)

                relations = [
                    r for r in relations if matches(getattr(r.properties, key))
                ]

        return RelationDataset(relations)


def get_relation_fn_type(relation_dict: dict) -> RelationFnType:
    """Determine the function type of a relation."""

    # Check if relation is one-to-many
    one_to_many = False
    sub2obj: dict[str, set[str]] = {}
    for sample in relation_dict["samples"]:
        cur = sub2obj.get(sample["subject"], set())
        cur.add(sample["object"])
        sub2obj[sample["subject"]] = cur
    for obj_set in sub2obj.values():
        if len(obj_set) > 1:
            one_to_many = True
            break

    # Check if relation is many-to-one
    many_to_one = False
    obj2sub: dict[str, set[str]] = {}
    for sample in relation_dict["samples"]:
        cur = obj2sub.get(sample["object"], set())
        cur.add(sample["subject"])
        obj2sub[sample["object"]] = cur
    for sub_set in obj2sub.values():
        if len(sub_set) > 1:
            many_to_one = True
            break

    # Determine relation type
    if one_to_many and many_to_one:
        return "MANY_TO_MANY"
    elif one_to_many:
        return "ONE_TO_MANY"
    elif many_to_one:
        return "MANY_TO_ONE"
    else:
        return "ONE_TO_ONE"


def load_relation_dict(file) -> dict:
    """Load dict for a single relation from a json file."""
    # file = Path(file)
    # if file.suffix != ".json":
    #     raise ValueError(f"relation files must be json, got: {file}")
    with open(file,"r") as handle:
        relation_dict = json.load(handle)
    for key in ("domain", "range"):
        if key in relation_dict:
            relation_dict[f"_{key}"] = relation_dict.pop(key)

    # Check that all keys are valid kwargs to Relation
    valid_keys = set(field.name for field in fields(Relation))
    for key in relation_dict.keys():
        if key not in valid_keys:
            raise ValueError(
                f"invalid key in relation file {file}: {key}. "
                f"valid keys are: {valid_keys}"
            )

    # Compute the type of relation function (injection, surjection, bijection, etc.)
    relation_dict["properties"]["fn_type"] = get_relation_fn_type(relation_dict)#就是看下数据类型

    return relation_dict


def load_relation(file) -> Relation:
    """Load a single relation from a json file."""
    return Relation.from_dict(load_relation_dict(file))


def load_dataset(*paths) -> RelationDataset:
    """Load relations from json files in a folder.

    Accepts one or more directories or files. If a file, should be JSON format, and will
    be read as one relation. If a directory, will recursively search for all JSON files.
    """
    # Load all relation files
    print(paths)
    files = []
    for path in paths:
        # path = Path(path)
        files.append(path)
        # if path.is_file():
        #     print("hello")
        #     logger.debug(f"found relation file: {path}")
        #     files.append(path)
        # else:
        #     print("fuck")
        #     logger.debug(f"{path} is directory, globbing for json files...")
        #     for file in sorted(path.glob("**/*.json")):
        #         logger.debug(f"found relation file: {file}")
        #         files.append(file)
    logger.debug(f"found {len(files)} relation files total, loading...")
    relation_dicts = [load_relation_dict(file) for file in files]
    # Mark all disambiguating relations
    domain_range_pairs: dict[tuple[str, str], int] = {}
    for relation_dict in relation_dicts:
        d, r = (
            relation_dict["properties"]["domain_name"],
            relation_dict["properties"]["range_name"],
        )
        cur = domain_range_pairs.get((d, r), 0)
        domain_range_pairs[(d, r)] = cur + 1

    for relation_dict in relation_dicts:
        d, r = (
            relation_dict["properties"]["domain_name"],
            relation_dict["properties"]["range_name"],
        )
        relation_dict["properties"]["disambiguating"] = domain_range_pairs[(d, r)] > 1

    # Create Relation objects
    relations = [Relation.from_dict(relation_dict) for relation_dict in relation_dicts]

    return RelationDataset(relations)

def patch(sentence: str):

    return sentence

class KnowledgeDataset:
    def __init__(
        self,
        N = 1,
        knowledge_type='factual/',
        relation_name='/city_in_country.json',
        tokenizer=None,
        prompts=None,
        data_path="./data/",
        prepend_bos=False,
        seed=None,
    ):
        """
        type:
            different knowledge type including: [factual, commonsense,linguistic]
        """
        assert seed is not None
        random.seed(seed)

        if not (
            N == 1
            or prepend_bos == False
            or tokenizer.bos_token_id == tokenizer.eos_token_id
        ):
            warnings.warn(
                "Probably word_idx will be calculated incorrectly due to this formatting"
            )
        
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        self.relation = load_dataset(data_path+knowledge_type+relation_name).relations[0]#TODO 这个地方也有问题，先跑起来再说吧


        prompt_template = self.relation.prompt_templates[0]#TODO 一个文件有2个模版我只选第一个
        self.sentences = [
            prompt_template.format(sample.subject) for sample in self.relation.samples
        ]  # a list of strings. Renamed as this should NOT be forward passed
        self.patch_sentences = [
            patch(prompt_template.format(sample.subject)) for sample in self.relation.samples
        ]
        self.answers = [
            sample.object for sample in self.relation.samples
        ]

        # valid_data, test_data = relation.split(N)
        self.toks = torch.Tensor(self.tokenizer(self.sentences, padding=True).input_ids).type(
            torch.int
        )

def get_and_filter_dataset(knowledge_type='factual/',
                           relation_name='city_in_country.json',
                           tokenizer=None,
                        #    model=None,
                           data_path="./data/",
                           ):
    relation = load_dataset(data_path+knowledge_type+relation_name)[0]
    prompt_template = relation.prompt_templates[0]
    sentences = [
        prompt_template.format(sample.subject) for sample in self.relation.samples
    ]  # a list of strings. Renamed as this should NOT be forward passed
    answers = [
        sample.object for sample in relation.samples
    ]
    toks = torch.Tensor(tokenizer(sentences, padding=True).input_ids).type(
            torch.int
        )
    answer_toks = torch.Tensor(tokenizer(answers, padding=True).input_ids).type(
            torch.int
        )
    return toks, answer_toks
