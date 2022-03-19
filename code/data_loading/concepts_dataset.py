from itertools import product
from pathlib import Path
from typing import Dict, List, Union, Tuple, Any

import numpy as np
import pandas as pd
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset as TorchDataset, DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer

from data_loading.dataset_utils import filter_dataset
from data_loading.steps_scheduler import StepsScheduler
from modeling.masking.mapper import ConceptsMapping
from modeling.masking.masker import LanguageMasker
from project_utils.constants import (UNK, SEP, MAX_SEQ_LEN, BATCH_SIZE, NUM_WORKERS)
from project_utils.functions import (concat_dicts, to_path, save_json, load_json,
                                     divide_to_batches)


class ConceptsDataset(TorchDataset):
    non_concept_columns = ['example_id', 'text', 'split', 'input_ids', 'attention_mask',
                            'mlm_text', 'mlm_input_ids', 'mlm_attention_mask', 'mlm_output_input_ids']
    def __init__(self,
                 # dataset args
                 dataset: Union[str, Path, Dict[str, List[Any]], Dataset],
                 language_masker: Union[LanguageMasker, str, Path] = None,
                 max_seq_len: int = None,
                 concept_columns: List[str] = None,
                 concepts_to_control: List[str] = None,
                 seed: int = None):
        if isinstance(dataset, (str, Path)):
            dataset = to_path(dataset)
            dataset = Dataset.load_from_disk(dataset)
        elif isinstance(dataset, dict):
            dataset = Dataset.from_dict(dataset)
        self.dataset = dataset
        assert isinstance(dataset, Dataset), "dataset should be an arrow Dataset object"
        assert 'example_id' in self.dataset.column_names, "'example_id' should be in dataset column names."
        assert 'text' in self.dataset.column_names, "'text' should be in dataset column names."
        assert 'split' in self.dataset.column_names, "'split' should be in dataset column names."
        assert len(set(self.dataset['example_id'])) == len(self.dataset['example_id']), "'example_id' column must " \
                                                                                        "not contain duplicates."
        self._init_columns_properties(concept_columns, concepts_to_control)

        concepts_values = {c: sorted(list(set(self.dataset[c]))) for c in self.concept_columns}
        if language_masker is None:
            concepts_mapping = ConceptsMapping(concepts_values)
            self.language_masker = LanguageMasker(seed=seed, concepts_mapping=concepts_mapping)
            self.language_masker_path = None
        else:
            assert isinstance(language_masker, (str, Path, LanguageMasker)), "`language_masker` should be a json path" \
                                                                             " or a LanguageMasker object"
            if isinstance(language_masker, (str, Path)):
                self.language_masker_path = language_masker
                self.language_masker = LanguageMasker.from_json(language_masker)
            else:
                self.language_masker_path = None
                self.language_masker = language_masker
            err_msg = "`concepts_to_control` contains a concept that `language_masker` doesn't have."
            assert len(self.concepts_to_control) <= len(self.language_masker.concepts), err_msg

        if max_seq_len is None:
            if 'input_ids' not in self.dataset.column_names:
                max_seq_len = MAX_SEQ_LEN
            else:
                max_seq_len = max([len(ids) for ids in self.dataset['input_ids']])
        self.language_masker.processor.max_seq_len = max_seq_len

        self.seed = seed
        self.rnp = np.random.RandomState(self.seed)
        self.example2id = {example_id: i for i, example_id in enumerate(self.dataset['example_id'])}

        # adding input_ids and attention_mask to the dataset
        if 'input_ids' not in self.dataset.column_names or 'attention_mask' not in self.dataset.column_names:
            self.tokenize_dataset(text_column='text', new_tokenized_columns_prefix='')
        elif 'input_ids' in self.dataset.column_names:
            # need to tokenize since not all the input_ids have length of self.max_seq_len
            if sum([1 for ids in self.dataset['input_ids'] if len(ids) != self.max_seq_len]) != 0:
                self.tokenize_dataset(text_column='text', new_tokenized_columns_prefix='')

    @staticmethod
    def extract_concept_columns(dataset: Dataset) -> List[str]:
        concept_columns = [c for c in dataset.column_names
                            if (c not in ConceptsDataset.non_concept_columns and
                                not c.startswith('new_') and not c.startswith('orientation_'))]
        return concept_columns

    def _init_columns_properties(self, concept_columns: List[str], concepts_to_control: List[str]):
        assert concept_columns is None or isinstance(concept_columns, list), "`concept_columns` must be None or list"
        assert concepts_to_control is None or isinstance(concepts_to_control, list), "`concepts_to_control` must be" \
                                                                                     " None or list"

        if concept_columns is None:
            concept_columns = self.extract_concept_columns(self.dataset)
        else:
            concept_columns = [c for c in concept_columns if c in self.dataset.column_names]
        self.concept_columns = concept_columns
        if concepts_to_control is None:
            concepts_to_control = []
        else:
            concepts_to_control = [c for c in concepts_to_control if c in self.concept_columns]
        self.concepts_to_control = concepts_to_control
        columns = [c for c in self.non_concept_columns if c not in ['text', 'split', 'mlm_text']]
        columns += self.concept_columns + [f'new_{c}' for c in self.concept_columns]
        columns += self.concept_columns + [f'orientation_{c}' for c in self.concept_columns]
        self.columns_for_get_item = columns

    @property
    def processor(self):
        return self.language_masker.processor

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def max_seq_len(self):
        return self.processor.max_seq_len

    @property
    def concepts_mapping(self):
        return self.language_masker.concepts_mapping

    def replace_tokenizer(self, tokenizer: Union[str, PreTrainedTokenizer], tokenize: bool = True):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.language_masker.processor.tokenizer = tokenizer
        if tokenize:
            self.tokenize_dataset(text_column='text', new_tokenized_columns_prefix='')
            if 'mlm_text' in self.dataset.column_names:
                self.tokenize_dataset(text_column='mlm_text', new_tokenized_columns_prefix='mlm_')

    def _get_kwargs(self) -> Dict:
        language_masker = self.language_masker if self.language_masker_path is None else self.language_masker_path
        kwargs = {'dataset': self.dataset,
                  'language_masker': language_masker,
                  'max_seq_len': self.max_seq_len,
                  'concept_columns': self.concept_columns,
                  'concepts_to_control': self.concepts_to_control,
                  'seed': self.seed}
        return kwargs

    def to_json(self, dir_path: Union[str, Path], new_language_masker_path: bool = True, **kwargs_to_add):
        dir_path = to_path(dir_path, make_parent_dir=True)
        json_path = dir_path / 'kwargs.json'
        dataset_path = dir_path / 'dataset'
        if self.language_masker_path is None or new_language_masker_path:
            self.language_masker_path = dir_path / 'language_masker.json'
            self.language_masker.to_json(self.language_masker_path)
        self.dataset.save_to_disk(dataset_path)
        json_dict = self._get_kwargs()
        json_dict['dataset'] = str(dataset_path)
        json_dict['language_masker'] = str(self.language_masker_path)
        json_dict.update(kwargs_to_add)
        save_json(json_dict, json_path)

    @classmethod
    def from_json(cls, dir_path: Union[str, Path], **dataset_kwargs):
        json_path = to_path(dir_path) / 'kwargs.json'
        data_kwargs = load_json(json_path)
        data_kwargs.update(dataset_kwargs)
        return cls(**data_kwargs)

    def filter_dataset(self, key_values_to_keep: List[Dict[str, List[str]]] = None,
                       example_ids_to_keep: List[str] = None,
                       sampling_seed: int = None,
                       columns: List[str] = None) -> Union[None, Dataset]:
        new_dataset = self.dataset
        if example_ids_to_keep is not None:
            example_ids_to_keep = [eid for eid in example_ids_to_keep if eid in self.example2id]
            ids = [self.example2id[eid] for eid in example_ids_to_keep]
            if len(ids) == 0:
                return None
            new_dataset = new_dataset.select(ids)
        if key_values_to_keep is not None:
            new_dataset = filter_dataset(self.dataset, key_values_to_keep, seed=sampling_seed, columns=columns)
            if new_dataset is None:
                return None
        if columns is not None and key_values_to_keep is None:
            new_dataset = Dataset.from_dict({c: new_dataset[c] for c in columns})
        return new_dataset

    def create_new_instance(self, key_values_to_keep: List[Dict[str, List[str]]] = None,
                            example_ids_to_keep: List[str] = None,
                            save_new_instance_to_dir: Union[str, Path] = None,
                            sampling_seed: int = None,
                            columns: List[str] = None,
                            **new_instance_kwargs) -> Union[None, "ConceptsDataset"]:
        new_dataset = self.filter_dataset(key_values_to_keep, example_ids_to_keep, sampling_seed, columns)
        if new_dataset is None:
            return None
        kwargs = self._get_kwargs()
        kwargs.update(new_instance_kwargs)
        kwargs['dataset'] = new_dataset
        new_instance = ConceptsDataset(**kwargs)
        if save_new_instance_to_dir is not None and not save_new_instance_to_dir:
            new_instance.to_json(save_new_instance_to_dir)
        return new_instance

    def create_new_instance_for_augmentations(self,
                                              key_values_to_keep: List[Dict[str, List[str]]] = None,
                                              example_ids_to_keep: List[str] = None,
                                              save_new_instance_to_dir: Union[str, Path] = None,
                                              sampling_seed: int = None,
                                              columns: List[str] = None,
                                              concepts_values_for_augmentations: Dict[str, List[str]] = None,
                                              n_augmentations_per_example: Union[int, str] = 1,
                                              include_unknown_orientation: bool = False,
                                              **new_instance_kwargs) -> "ConceptsDataset":
        nape = n_augmentations_per_example
        assert (isinstance(nape, str) and nape == 'all') or (isinstance(nape, int) and nape > 0)
        rnp = np.random.RandomState(sampling_seed if sampling_seed is not None else self.seed)
        columns = columns if columns is not None else self.dataset.column_names
        new_dataset = self.filter_dataset(key_values_to_keep, example_ids_to_keep, sampling_seed, columns)
        if concepts_values_for_augmentations is None:
            concepts_values = self.language_masker.concepts_mapping.concepts_values
            concepts_values_for_augmentations = {c: vs for c, vs in concepts_values.items()
                                                 if c in self.concepts_to_control}
        if not include_unknown_orientation:
            concepts_values_for_augmentations = {c: [v for v in vs if v != UNK]
                                                 for c, vs in concepts_values_for_augmentations.items()}
        n_examples = len(new_dataset)
        if isinstance(nape, str) and nape == 'all':
            concepts = list(concepts_values_for_augmentations.keys())
            possible_orientations = {c: [] for c in concepts}
            for c, vs in concepts_values_for_augmentations.items():
                for v in vs:
                    possible_orientations[c] += self.language_masker.concept_value_ordered_orientations(c, v)
            combinations = list(product(*possible_orientations.values()))
            orientations = {c: [o[i] for o in combinations] * n_examples for i, c in enumerate(concepts)}
            new_concepts_values = {c: [self.language_masker.orientation_to_value(c, o) for o in os]
                                   for c, os in orientations.items()}
            nape = len(combinations)
        else:
            new_concepts_values = {c: rnp.choice(v, n_examples * nape).tolist()
                                   for c, v in concepts_values_for_augmentations.items()}
            empty_texts = ['' for _ in range(n_examples * nape)]
            orientations = self.sample_orientations(empty_texts, new_concepts_values, only_related=False)
        new_dataset = new_dataset.select(np.repeat(np.arange(0, n_examples), nape))

        indices = list(range(0, len(new_dataset)))
        batches = divide_to_batches(indices, 5000)
        function_returns = []
        for batch in tqdm(batches):
            dataset_batch = new_dataset[batch]
            texts = dataset_batch['text']
            concepts_values = {c: dataset_batch[c] for c in self.concept_columns}
            new_concepts_values_batch = {c: [vs[i] for i in batch] for c, vs in new_concepts_values.items()}
            orientations_batch = {c: [os[i] for i in batch] for c, os in orientations.items()}
            masked_data = self.language_masker.words_concepts_modeling_mask(texts, concepts_values,
                                                                            new_concepts_values_batch,
                                                                            noise=0.0,
                                                                            using_cached=True,
                                                                            return_tensors='np', device=None)
            example_ids = [f'{eid}{SEP}{i}' for i, eid in zip(batch, dataset_batch['example_id'])]
            to_add = {'example_id': example_ids,
                      'mlm_text': masked_data['masked_text'],
                      'mlm_input_ids': masked_data['mlm_input_ids'],
                      'mlm_attention_mask': masked_data['mlm_attention_mask'],
                      'mlm_output_input_ids': masked_data['mlm_output_input_ids']}
            to_add.update({f'new_{c}': o for c, o in new_concepts_values_batch.items()})
            to_add.update({f'orientation_{c}': o for c, o in orientations_batch.items()})
            function_returns.append(to_add)
        function_returns = concat_dicts(function_returns)
        new_dataset = {k: new_dataset[k] for k in new_dataset.column_names}
        new_dataset.update(function_returns)
        columns = columns + [c for c in new_dataset.keys() if c not in columns]
        new_dataset = Dataset.from_dict({c: new_dataset[c] for c in columns})
        kwargs = self._get_kwargs()
        if self.language_masker_path is not None:
            kwargs['language_masker'] = self.language_masker_path
        kwargs.update(new_instance_kwargs)
        kwargs['dataset'] = new_dataset
        new_instance = ConceptsDataset(**kwargs)
        if save_new_instance_to_dir is not None and not save_new_instance_to_dir:
            new_instance.to_json(save_new_instance_to_dir)
        return new_instance

    def get_indices(self) -> List[int]:
        return list(range(len(self.dataset)))

    def tokenize_dataset(self, text_column: str = 'text', new_tokenized_columns_prefix: str = ''):
        indices = self.get_indices()
        batches = divide_to_batches(indices, 500)
        function_returns = []
        for batch in tqdm(batches):
            batch_texts = self.dataset[batch][text_column]
            tokenized_data = self.processor.tokenize_texts(batch_texts, return_tensors='np',
                                                           padding='max_length', truncation=True,
                                                           max_length=self.max_seq_len, return_attention_mask=True)
            function_returns.append({f'{new_tokenized_columns_prefix}input_ids': tokenized_data['input_ids'],
                                     f'{new_tokenized_columns_prefix}attention_mask': tokenized_data['attention_mask']})
        function_returns = concat_dicts(function_returns)
        dataset = {k: self.dataset[k] for k in self.dataset.column_names}
        dataset.update(function_returns)
        self.dataset = Dataset.from_dict(dataset)

    def add_mlm_masks(self, text_column: str = 'text',
                      orientations: Dict[str, List[str]] = None,
                      mode: str = None):
        if self.language_masker is None:
            raise ValueError("Cannot call self.add_mlm_masks() when self.language_masker is None.")
        indices = self.get_indices()
        if mode is None:
            indices_split = pd.DataFrame({'indices': indices, 'split': self.dataset['split']})
            split_batches = [gb[1].tolist() for gb in indices_split.groupby('split')['indices']]
            batches = []
            for split_batch in split_batches:
                batches += divide_to_batches(split_batch, 5000)
        else:
            batches = divide_to_batches(indices, 5000)
        function_returns = []
        for batch in tqdm(batches):
            dataset_batch = self.dataset[batch]
            assert len(set(dataset_batch['split'])) == 1
            split = dataset_batch['split'][0] if mode is None else mode
            texts = dataset_batch[text_column]
            concepts_values = {c: dataset_batch[c] for c in self.concept_columns}
            concepts_values_to_control = {c: v for c, v in concepts_values.items() if c in self.concepts_to_control}
            if orientations is not None:
                batch_orientations = {}
                new_concepts_values = {}
                for c in concepts_values_to_control:
                    batch_orientations[c] = [orientations[c][i] for i in batch]
                    new_concepts_values[c] = [self.language_masker.orientation_to_value(c, o)
                                              for o in batch_orientations[c]]
                masked_data = self.language_masker.words_concepts_modeling_mask(texts, concepts_values,
                                                                                new_concepts_values,
                                                                                using_cached=True,
                                                                                return_tensors='np', device=None)
            else:
                new_concepts_values = self.sample_new_values(concepts_values_to_control)
                masked_data = self.language_masker.words_concepts_modeling_mask(texts, concepts_values,
                                                                                new_concepts_values,
                                                                                using_cached=True,
                                                                                return_tensors='np', device=None)
                if split in ['train', 'unlabeled']:
                    new_concepts_values = concepts_values_to_control
                    only_related = True
                else:
                    only_related = False
                batch_orientations = self.sample_orientations(texts, new_concepts_values,
                                                              only_related=only_related)
            to_add = {'index': batch,
                      'mlm_text': masked_data['masked_text'],
                      'mlm_input_ids': masked_data['mlm_input_ids'],
                      'mlm_attention_mask': masked_data['mlm_attention_mask'],
                      'mlm_output_input_ids': masked_data['mlm_output_input_ids']}
            to_add.update({f'new_{c}': o for c, o in new_concepts_values.items()})
            to_add.update({f'orientation_{c}': o for c, o in batch_orientations.items()})
            function_returns.append(to_add)
        function_returns = concat_dicts(function_returns)
        dataset = {k: self.dataset[k] for k in self.dataset.column_names}
        # need to reorder the returns list
        unsorted_indices = function_returns.pop('index')
        for k, v in function_returns.items():
            sorted_v = [None] * len(indices)
            for i, value in zip(unsorted_indices, v):
                sorted_v[i] = value
            dataset[k] = sorted_v
        self.dataset = Dataset.from_dict(dataset)

    def sample_new_values(self, concepts_values: Dict[str, List[str]], seed: int = None):
        rnp = np.random.RandomState(seed) if seed is not None else self.rnp
        new_values = {}
        # we need to sample new concepts-values so the real concept-values related words will be masked.
        for c in concepts_values:
            new_values[c] = []
            for cv in concepts_values[c]:
                values = [v for v in self.language_masker.concepts_mapping.concept_values(c)
                          if v not in [cv, UNK]]
                value = UNK if len(values) == 0 else rnp.choice(values, 1).tolist()[0]
                new_values[c].append(value)
        return new_values

    def sample_orientations(self, texts: List[str],
                            new_concepts_values: Dict[str, List[str]],
                            only_related: bool = True,
                            seed: int = None) -> Dict[str, List[str]]:
        rnp = np.random.RandomState(seed) if seed is not None else self.rnp
        orientations = {c: [] for c in new_concepts_values.keys()}
        for i, text in enumerate(texts):
            if only_related:
                text_words = self.processor.get_words(text, n_grams=1, stem=True) if only_related else None
            else:
                text_words = []
            for concept in new_concepts_values:
                new_value = new_concepts_values[concept][i]
                if new_value == UNK:
                    orientations[concept].append(UNK)
                    continue
                vos = self.language_masker.concept_value_ordered_orientations(concept, new_value)
                if only_related:
                    # we always keep the first orientation since it is the main / value orientation
                    vos = [vos[0]] + [w for w in vos[1:] if w in text_words]
                if len(vos) == 1:
                    orientations[concept].append(vos[0])
                else:
                    orientations[concept].append(rnp.choice(vos, size=1).tolist()[0])
        return orientations

    def refresh(self):
        self.add_mlm_masks('text')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        # need to convert otherwise data contains lists and there is an "OSError: [Errno 24] Too many open files"
        data = {k: np.array(v) if isinstance(v, list) else v for k, v in data.items()
                if k in self.columns_for_get_item}
        return data

    def add_new_concepts_values(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        new_concepts_values = {}
        for concept in batch['orientations']:
            new_concepts_values[concept] = []
            concept_values = batch['concepts_values'][concept]
            orientations = batch['orientations'][concept]
            for cv, o in zip(concept_values, orientations):
                ov = self.language_masker.orientation_to_value(concept, o)
                new_value = ov if ov != UNK else cv  # in the orientation is UNK, we assume the concept is unchanged
                new_concepts_values[concept].append(new_value)
        batch['new_concepts_values'] = new_concepts_values
        return batch

    def fix_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch['concepts_values'] = {c: batch[c] for c in self.concept_columns}
        batch['orientations'] = {}
        for c in self.concepts_to_control:
            if f'orientation_{c}' not in batch:
                batch[f'orientation_{c}'] = [UNK for _ in batch[c]]
        batch['orientations'] = {c: batch[f'orientation_{c}'] for c in self.concepts_to_control}
        batch = self.add_new_concepts_values(batch)
        return batch


class ConceptsDataModule(LightningDataModule):
    def __init__(self,
                 train_dataset: ConceptsDataset,
                 val_dataset: ConceptsDataset,
                 test_dataset: ConceptsDataset,
                 batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_WORKERS,
                 seed: int = None,
                 evaluation_step: str = 'generation',
                 refresh_mlm: bool = False,
                 refresh_train_dataset_every_n_epochs: int = None,
                 # step args
                 start_probabilities: Union[Tuple, Dict[str, float]] = (('mlm', 1.0), ('classifier', 0.0)),
                 end_probabilities: Union[Tuple, Dict[str, float]] = None,
                 n_steps: int = 1,
                 unknown_orientation_p: float = 0.0,
                 # generate kwargs
                 generate_kwargs: Dict[str, Any] = None,
                 **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.evaluation_step = evaluation_step
        self.refresh_mlm = refresh_mlm
        self.refresh_train_dataset_every_n_epochs = refresh_train_dataset_every_n_epochs
        self.epochs = 0
        self.scheduler = StepsScheduler(start_probabilities, end_probabilities, n_steps, seed)
        self.unknown_orientation_p = unknown_orientation_p
        self.generate_kwargs = generate_kwargs if generate_kwargs is not None else {}

    @property
    def rnp(self):
        return self.scheduler.rnp

    @classmethod
    def from_concepts_dataset(cls, concepts_dataset: ConceptsDataset,
                              train_settings: List[Dict[str, Any]] = None,
                              val_settings: List[Dict[str, Any]] = None,
                              test_settings: List[Dict[str, Any]] = None,
                              sampling_seed: int = None,
                              **kwargs):
        train_dataset = concepts_dataset.create_new_instance(key_values_to_keep=train_settings,
                                                             sampling_seed=sampling_seed)
        val_dataset = concepts_dataset.create_new_instance(key_values_to_keep=val_settings,
                                                           sampling_seed=sampling_seed)
        test_dataset = concepts_dataset.create_new_instance(key_values_to_keep=test_settings,
                                                            sampling_seed=sampling_seed)
        return cls(train_dataset, val_dataset, test_dataset, **kwargs)

    def update_probabilities(self, start_probabilities: Union[Tuple, Dict[str, float]] = None,
                             end_probabilities: Union[Tuple, Dict[str, float]] = (('mlm', 1.0),),
                             n_steps: int = 1, seed: int = None):
        self.scheduler = StepsScheduler(start_probabilities, end_probabilities, n_steps, seed)

    def prepare_batch(self, batch: Dict[str, Any], dataset: ConceptsDataset, step_type: str):
        batch = dataset.fix_batch(batch)
        if step_type == 'classifier':
            batch['ignore_unknown'] = True
            batch['with_encoder_grad'] = True
            batch['step_type'] = 'classifier'
        elif step_type == 'mlm':
            for c in batch['orientations']:
                co = batch['orientations'][c]
                unknown_indices = self.rnp.uniform(0, 1, len(co)) <= self.unknown_orientation_p
                batch['orientations'][c] = [UNK if unknown_indices[i] else o for i, o in enumerate(co)]
            batch = dataset.add_new_concepts_values(batch)
            batch['ignore_unknown'] = True
            batch['step_type'] = 'mlm'
        elif step_type == 'generation':
            batch['ignore_unknown'] = True
            batch['step_type'] = 'generation'
            for k, v in self.generate_kwargs.items():
                batch[k] = v
        else:
            raise ValueError("Unknown step type")
        return batch

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            step_type = self.scheduler.sample_step_type(update=True)
            batch = self.prepare_batch(batch, self.train_dataset, step_type)
        elif self.trainer.evaluating:
            batch = self.prepare_batch(batch, self.val_dataset, self.evaluation_step)
        else:  # self.trainer.testing
            batch = self.prepare_batch(batch, self.test_dataset, self.evaluation_step)
        return batch

    def train_dataloader(self) -> DataLoader:
        if (self.refresh_train_dataset_every_n_epochs is not None
                and self.refresh_train_dataset_every_n_epochs > 0
                and self.epochs > 0
                and self.epochs % self.refresh_train_dataset_every_n_epochs == 0):
            if self.refresh_mlm:
                self.train_dataset.refresh()
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.num_workers)
        self.epochs += 1
        return dataloader

    def val_dataloader(self) -> DataLoader:
        batch_size = int(self.batch_size * 1.5)
        num_return_sequences = self.generate_kwargs.get('num_return_sequences', 1)
        if self.evaluation_step == 'generation' and num_return_sequences > 1:
            batch_size = int(batch_size / num_return_sequences)
        return DataLoader(self.val_dataset, batch_size=batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        batch_size = int(self.batch_size * 1.5)
        num_return_sequences = self.generate_kwargs.get('num_return_sequences', 1)
        if self.evaluation_step == 'generation' and num_return_sequences > 1:
            batch_size = int(batch_size / num_return_sequences)
        return DataLoader(self.test_dataset, batch_size=batch_size, num_workers=self.num_workers)
