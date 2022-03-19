import json
import os
import pickle
from collections import defaultdict
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Union, Dict, List, Any, Tuple

import numpy as np
import torch
from torch import cuda
from torch.nn import functional as nn_functional
from transformers import AdamW


def divide_to_batches(list_to_divide: List, batch_size: int):
    return [list_to_divide[i * batch_size: (i + 1) * batch_size]
            for i in range(int(np.ceil(len(list_to_divide) / batch_size)))]


def count_num_cpu_gpu() -> Tuple[int, int]:
    num_cpu_cores = cpu_count() // 2
    if cuda.is_available():
        num_gpu_cores = cuda.device_count()
        num_cpu_cores = num_cpu_cores // num_gpu_cores
    else:
        num_gpu_cores = 0
    return num_cpu_cores, num_gpu_cores


def to_path(path: Union[str, Path], make_parent_dir: bool = False):
    if isinstance(path, str):
        path = Path(path)
    path = path.resolve()
    if make_parent_dir:
        os.makedirs(path.parent, exist_ok=True)
    return path


def save_pkl(to_save: Any, file_path: Union[str, Path]):
    file_path = to_path(file_path, make_parent_dir=True)
    with open(file_path, 'wb') as f:
        pickle.dump(to_save, f)


def load_pkl(file_path: Union[str, Path]):
    file_path = to_path(file_path)
    with open(file_path, 'rb') as f:
        instance = pickle.load(f)
        return instance


def save_json(to_save: Any, file_path: Union[str, Path]):
    file_path = to_path(file_path, make_parent_dir=True)
    with open(file_path, 'w') as f:
        json.dump(to_save, f)


def load_json(file_path: Union[str, Path]):
    file_path = to_path(file_path)
    with open(file_path, 'r') as f:
        instance = json.load(f)
        return instance


def subset_dict(source_dict: Dict, wanted_keys: List):
    wanted_keys = [k for k in wanted_keys if k in source_dict]
    return {k: source_dict[k] for k in wanted_keys}


def concat_dicts(dicts: List[Dict], stack: bool = False) -> Dict[Any, Union[List, np.ndarray]]:
    new_dict = defaultdict(list)
    for dic in dicts:
        for k, v in dic.items():
            if isinstance(v, list):
                new_dict[k].extend(v)
            elif isinstance(v, np.ndarray):
                if stack:
                    v = v.reshape((1, -1))
                new_dict[k].append(v)
            else:
                new_dict[k].append(v)
    d = dict(new_dict)
    return {k: v if not isinstance(v[0], np.ndarray) else np.concatenate(v, axis=0) for k, v in d.items()}


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = nn_functional.normalize(a, p=2, dim=1)
    b_norm = nn_functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def change_dtype_precision(torch_object: Union[torch.Tensor, torch.nn.Module], precision: int):
    if precision == 16:
        return torch_object.half()
    elif precision == 32:
        return torch_object.float()
    else:
        return torch_object.double()


def calculate_total_training_steps(train_len: int, batch_size: int,
                                   accumulate_grad_batches: int, gpus: int, max_epochs: int) -> int:
    return int((train_len // (batch_size * float(max(1, gpus)))) // accumulate_grad_batches * max_epochs)


def configure_adamw_with_decay(model: torch.nn.Module,
                               weight_decay: float = None,
                               lr: float = None,
                               eps: float = None):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)


def create_experiment_name_and_dir(output_dir: Union[Path, str], experiment_name: str = '',
                                   add_datetime: bool = False):
    if add_datetime:
        experiment_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
    experiment_dir = to_path(output_dir) / experiment_name
    os.makedirs(experiment_dir, exist_ok=True)
    experiment_dir.mkdir(exist_ok=True, parents=True)
    experiment_dir = str(experiment_dir)
    return experiment_name, experiment_dir


def sequence_indices(primary_list: List[str], sequence: List[Any]) -> List[List[int]]:
    indices = []
    len_primary = len(primary_list)
    len_seq = len(sequence)
    for i in range(len_primary):
        if primary_list[i:i + len_seq] == sequence:
            indices.append(list(range(i, i + len_seq)))
    return indices


def repeat_lists_in_dict(dict_with_lists: Dict[str, Any], repeat: int):
    for k, v in dict_with_lists.items():
        if isinstance(v, list):
            dict_with_lists[k] = np.repeat(np.array(v, dtype='object'), repeat).tolist()
    return dict_with_lists


def round_power_two(x: int) -> int:
    power2 = 2**((x-1).bit_length()-1)
    diff2 = 2**((x-power2-1).bit_length()-1)
    return power2 + diff2