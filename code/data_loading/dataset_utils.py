from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from datasets import concatenate_datasets, Dataset, DatasetDict
from transformers import T5TokenizerFast

from modeling.masking.text_processor import TextProcessor
from project_utils.constants import UNK
from project_utils.functions import to_path


def load_dataset(dataset_path: Union[str, Path]) -> Dataset:
    dataset_path = to_path(dataset_path)
    return Dataset.load_from_disk(dataset_path)


def load_datasets(datasets_path: Union[str, Path]) -> DatasetDict:
    datasets_path = to_path(datasets_path)
    return DatasetDict.load_from_disk(str(datasets_path))


def concat_datasets(datasets: Union[str, Path, DatasetDict, Dict[str, Dataset]],
                    domains: Union[str, List[str]] = None,
                    splits: Union[str, List[str]] = None) -> Dataset:
    if isinstance(datasets, (str, Path)):
        datasets = load_datasets(datasets)
    if isinstance(domains, str):
        domains = [domains]
    if isinstance(splits, str):
        splits = [splits]
    to_concat = []
    for key, dataset in datasets.items():
        domain, split = key.split('_')
        if (domains is None or domain in domains) and (splits is None or split in splits):
            to_concat.append(dataset)
    return concatenate_datasets(to_concat)


def create_filter_mask(dataset: Dataset,
                       key_values_to_keep: List[Dict[str, List[str]]],
                       seed: int = None):
    true_mask = pd.Series([True for _ in range(len(dataset[list(dataset.column_names)[0]]))])
    mask = (true_mask == False).copy()
    for kv in key_values_to_keep:
        n_samples = kv.get('n_samples', None)
        inner_mask = true_mask.copy()
        for key, values in kv.items():
            if key == 'n_samples':
                continue
            key_mask = pd.Series(dataset[key]).isin(values)
            inner_mask = inner_mask & key_mask
        if n_samples is not None:
            true_indices = inner_mask[inner_mask == True]
            if n_samples < 1.0:
                true_indices = true_indices.sample(frac=n_samples, replace=False, random_state=seed)
            else:
                n_samples = int(min(n_samples, len(true_indices)))
                true_indices = true_indices.sample(n=n_samples, replace=False, random_state=seed)
            inner_mask = inner_mask & inner_mask.index.isin(true_indices.index.tolist())
        mask = mask | inner_mask
    return mask[mask].index.tolist()


def filter_dataset(dataset: Dataset,
                   key_values_to_keep: List[Dict[str, List[str]]],
                   columns: List[str] = None,
                   seed: int = None) -> Union[None, Dataset]:
    mask = create_filter_mask(dataset, key_values_to_keep, seed)
    if len(mask) == 0:
        return None
    filtered_dataset = dataset.select(mask)
    if columns is not None:
        filtered_dataset = Dataset.from_dict({c: filtered_dataset[c] for c in columns})
    return filtered_dataset


def fix_text_airline_domain(text: str):
    splitted = text.split('.')
    if len(splitted) > 1 and len(splitted[1]) > 0:
        splitted = splitted[1:]
    text = '.'.join(splitted).strip()
    return text