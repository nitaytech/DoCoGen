import os
from pathlib import Path
from typing import Union, List, Any, Tuple

from copy import copy
from inspect import signature

from project_utils.constants import (PRECISION, T5_MODEL_NAME, MAX_SEQ_LEN, BATCH_SIZE, NUM_WORKERS)
from project_utils.functions import to_path, load_json, save_json
from modeling.masking.text_processor import TextProcessor


class Configs:
    def __init__(self,
                 # project kwargs
                 project_dir: Union[str, Path],
                 raw_data_json_path: Union[str, Path],

                 concept_to_control: str,
                 values_to_control: List[str],
                 splits_for_training: Union[List[str], Tuple] = ('unlabeled',),
                 splits_for_augmentations: Union[List[str], Tuple] = ('train', 'validation', 'test'),

                 t5_model_name: str = T5_MODEL_NAME,
                 max_seq_len: int = MAX_SEQ_LEN,
                 seed: int = 42,
                 batch_size: int = BATCH_SIZE,

                 # preprocessing kwargs
                 min_occurrences: int = 10,
                 smoothing: Union[List[int], Tuple, str] = (1, 5, 7),
                 n_orientations: int = 4,
                 top_occurrences_threshold: int = 100,

                 # language_masker kwargs
                 n_grams: int = 3,
                 masked_output: bool = True,
                 threshold: float = 0.08,
                 top_n: float = 0.05,
                 noise: float = 0.05,

                 # concepts_data_module kwargs
                 num_workers: int = NUM_WORKERS,
                 refresh_mlm: bool = True,
                 refresh_train_dataset_every_n_epochs: int = 5,
                 unknown_orientation_p: float = 0.05,

                 # experiments kwargs
                 generator_epochs: int = 5,
                 generator_classifier_epochs: int = 3,
                 generator_classifier_batch_size: int = 64,
                 save_to_csv: bool = True,
                 save_to_csv_every_n_step: int = 500,

                 # generation kwargs
                 num_beams: int = 8,
                 num_beam_groups: int = 4,
                 repetition_penalty: float = 2.0,
                 length_penalty: float = 1.5,
                 diversity_penalty: float = 0.05,
                 early_stopping: bool = True,
                 num_return_sequences: int = 1,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: int = 40,
                 top_p: float = 1.0,

                 # optimizer kwargs
                 optimizer_weight_decay: float = 1e-5,
                 optimizer_lr: float = 5e-5,
                 optimizer_eps: float = 1e-8,
                 warmup_steps: int = 0,

                 # trainer kwargs
                 accumulate_grad_batches: int = 1,
                 amp_level: str = '01',
                 gpus: int = 1,
                 gradient_clip_val: float = 1.0,
                 log_every_n_steps: int = 500,

                 # path kwargs
                 data_dir: Union[str, Path] = None,
                 models_dir: Union[str, Path] = None,
                 language_masker_path: Union[str, Path] = None,
                 training_dataset_path: Union[str, Path] = None,
                 augmentations_dataset_path: Union[str, Path] = None,
                 trained_generator_ckpt: Union[str, Path] = None,
                 trained_generator_classifier_ckpt: Union[str, Path] = None,
                 *, fast_dev: bool = False):
        """
         ----- project kwargs -----
         project_dir: a path-like to the directory of the projec, where all the data and models will be saved.
         raw_data_json_path: a path-like to the json file containing the dataset. The keys should be the column
            names and the values are lists (of the same size). The dataset must containg the following keys:
            'split', 'text', (optional - 'example_id'), the concept you wish to control and (optional - other concepts).
            If 'example_id' is not provided, a unique identifier will be given to each example
            in the format of f'{split}_{i}'.

         concept_to_control: a string of the column/key you wish to control (concept is also known as attribute
            in controlable generation, in the DoCoGen paper - the concept is 'domain').
            values_to_control: a list of strings. Each string is a possible value of the controlled concept. These are
            the only values you wish to control. For example: ['airline', 'kitchen'],
         splits_for_training: a list or tuple of strings [default: ('unlabeled',)]. This are the the data splits
            (as given in the 'split' column) which will be used for training the masker and the generator.
         splits_for_augmentations: a list or tuple of strings [default: ('train', 'validation', 'test')]. This are
            the the data splits (as given in the 'split' column) which will be used as inpups, for generating new examples.

         t5_model_name: a string [default: 't5-base'], can be: 't5-small', 't5-base', 't5-large'.
         max_seq_len: an int [default: 96]. The maximal size of the inputs or the generated texts (examples longer
            than `max_seq_len` will be truncated).
         seed: an int [default: 42]. The random seed.

         ----- masker kwargs -----
         min_occurrences: an int [default: 10]. Terms (n-grams) that appear in less than `min_occurrences` examples
            will have a uniform distribution over the controlled values ($P(D|w)=1/N$) and the LanguageMasker object
            won't store their probabilities.
         smoothing: a string, or a list or tuple of ints [default: (1, 5, 7)]. The smoothing hyperparameter $a$
            which is used when estimating the probability of the controlled value $D$ given the term $w$:
            $P(D|w)=#_{w|D}+a/n_D$. If a string is given it should be 'n', in that case the smoothing hyperparameter
            will be eual to the number of words in the term. If a list or tuple of ints is given, it should be as the
            size of `n_grams`, and entry i corresponds to the smoothing hyperparameter used for an n-gram of size i.
            For example if `smoothing_hyperparameter =(1, 3, 7)` then 1, 3 and 7 are the smoothing hyperparameters
            of uni-grams, bi-grams and tri-grams respectively.
         n_orientations: an int [default: 4]. The number of orientations for each value to control (if there are
            5 values to control, there will be `5*n_orientations` orientations (including 5 orientations initialized
            with the value name).  The orientations are the top representing words of each controlled value $D$:
            words which reach the highest score of $log(#_{w|D}+ 1)P(w,D)$
         top_occurrences_threshold: an int [default: 100]. Words which appear in less than `top_occurrences_threshold`
            won't be considered as a possible orientation.
         n_grams: an int [default: 3], the maximum number of words in a term (n-gram).
         masked_output: a bool [default: True]. The generator (T5) can be trained to generate a only the masked spans)
            or to whole output. For example, given the input 'I <extra_id_0> you', the output can be either '<extra_id_0> love <extra_id_1>' or 'I love you'.
         threshold: a float [default: 0.08]. Terms (n-grams) with a masking score above `threshold` are masked.
            The masking score of a term with an origin value $D$ and a destination value $D'$ is
            $(1-H(D|w)/logN)(P(D|w)-P(D'|w)$.
         top_n: a float [default: 0.05]. A fraction of words with the highest masking score which are masked
            (including terms with a masking score above the threshold).
         noise: a float [default: 0.05]. A fraction of additional randomly masked words while training the generator.

         ----- data_module kwargs -----
         num_workers: an int [default: NUM_WORKERS]
         refresh_mlm: a bool [default: True], if True, masking again all the examples in the dataset every
            `refresh_train_dataset_every_n_epochs` epochs, by randomly sampling a destination value.
         refresh_train_dataset_every_n_epochs: an int [default: 5]. If `refresh_train_dataset_every_n_epochs` is 0
            or None, then the dataset won't be refreshed.
         unknown_orientation_p: a float [default: 0.05], the probability of sampling an UNK ('Unknown') orientation
            while training the generator (used for adding noise).

         ----- experiments kwargs -----
         batch_size: an int [default: 48]. The batch size of the the input to the generator.
         generator_epochs: an int [default: 5]. Number of epochs for training the generator.
         generator_classifier_batch_size: an int [default: 64]. The batch size of the the input to the generator's classifier.
         generator_classifier_epochs: an int [default: 3]. Number of epochs for training the generator's classifier, trained to predict the values of the controlled concept. This classifier is used for selecting the best version of the generator, by measuring the accuracy of the generated examples (during each epoch's validation step).
         save_to_csv: a bool [default: True] If True saving a csv file to the model's directory with metrics (model performances) at each epoch (file name: 'metrics.csv').
         save_to_csv_every_n_step: an int [default: 500] If `save_to_csv = True`, saving a csv file to the model's directory with training metrics every `save_to_csv_every_n_step` steps (file name: 'training_results.csv').

         ----- generation kwargs -----
         see: https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate
         num_beams: an int [default: 4]
         num_beam_groups: an int [default: 4]
         repetition_penalty: a float [default: 1.5]
         length_penalty: a float [default: 2.0]
         diversity_penalty: a float [default: 0.05]
         early_stopping: a bool [default: True]
         num_return_sequences: an int [default: 1]
         do_sample: a bool [default: False]
         temperature: a float [default: 1.0]
         top_k: an int [default: 40]
         top_p: a float [default: 1.0]

         ----- optimizer and scheduler kwargs -----
         see: https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/optimizer_schedules#transformers.AdamW
         optimizer_weight_decay: a float [default: 1e-5]
         optimizer_lr: a float [default: 5e-5]
         optimizer_eps: a float [default: 1e-8]
         warmup_steps: an int [default: 0]

         ----- trainer kwargs -----
         see: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags
         accumulate_grad_batches: an int [default: 1]
         amp_level: a string [default: '01']
         gpus: an int [default: 1]
         gradient_clip_val: a float [default: 1.0]
         log_every_n_steps: an int [default: 500]

         ----- path kwargs -----
         These paths can be provided, otherwise the paths will be automatically generated (`project_dir` is used as a prefix).
         data_dir: a path-like [default: None] a folder containing the datasets and the masker.
         models_dir: a path-like [default: None] a folder containing the generator and the generator's classifier.
         language_masker_path: a path-like [default: None]
         training_dataset_path: a path-like [default: None]
         augmentations_dataset_path: a path-like [default: None]
         trained_generator_ckpt: a path-like [default: None]
         trained_generator_classifier_ckpt: a path-like [default: None]
        """
        self.concept_to_control = concept_to_control
        self.values_to_control = values_to_control
        self.splits_for_training = list(splits_for_training)
        self.splits_for_augmentations = list(splits_for_augmentations)
        self.t5_model_name = t5_model_name
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.batch_size = batch_size
        self.min_occurrences = min_occurrences
        self.smoothing = smoothing
        self.n_orientations = n_orientations
        self.top_occurrences_threshold = top_occurrences_threshold
        self.n_grams = n_grams
        self.masked_output = masked_output
        self.threshold = threshold
        self.top_n = top_n
        self.noise = noise
        self.num_workers = num_workers
        self.refresh_mlm = refresh_mlm
        self.refresh_train_dataset_every_n_epochs = refresh_train_dataset_every_n_epochs
        self.unknown_orientation_p = unknown_orientation_p
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.diversity_penalty = diversity_penalty
        self.early_stopping = early_stopping
        self.num_return_sequences = num_return_sequences
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_lr = optimizer_lr
        self.optimizer_eps = optimizer_eps
        self.warmup_steps = warmup_steps
        self.accumulate_grad_batches = accumulate_grad_batches
        self.amp_level = amp_level
        self.gpus = gpus
        self.gradient_clip_val = gradient_clip_val
        self.log_every_n_steps = log_every_n_steps
        self.save_to_csv = save_to_csv
        self.save_to_csv_every_n_step = save_to_csv_every_n_step
        self.generator_epochs = generator_epochs
        self.generator_classifier_epochs = generator_classifier_epochs
        self.generator_classifier_batch_size = generator_classifier_batch_size
        self.fast_dev = fast_dev
        self.concepts_to_control = [concept_to_control]
        self.encoder_model_name = t5_model_name
        self.precision = PRECISION
        self.fast_dev_run = False
        self.text_processor = TextProcessor(**self.kwargs_for_class(TextProcessor))

        self._init_paths(project_dir,
                         raw_data_json_path,
                         data_dir=data_dir,
                         models_dir=models_dir,
                         language_masker_path=language_masker_path,
                         training_dataset_path=training_dataset_path,
                         augmentations_dataset_path=augmentations_dataset_path,
                         trained_generator_ckpt=trained_generator_ckpt,
                         trained_generator_classifier_ckpt=trained_generator_classifier_ckpt)
        # attributes that need to be updated before running an experiment
        self.max_epochs = None
        self.monitor = None
        self.mode = None
        self.start_probabilities = None
        self.end_probabilities = None
        self.n_steps = None
        self.output_dir = None
        self.language_masker = self.language_masker_path
        self.concepts_mapping = None
        self.concepts_to_predict = None
        self.ignore_unknown = None
        self.adversarial_concepts = None
        self.training_steps = None
        self.evaluation_step = None

    def _init_paths(self, project_dir: Union[str, Path],
                    raw_data_json_path: Union[str, Path],
                    **kwargs):
        def kwargs_get(k: str, default: Any):
            v = kwargs.get(k, None)
            if v is None:
                v = default
            if v is None:
                return None
            else:
                return str(v)

        project_dir = to_path(project_dir)

        data_dir = to_path(kwargs_get('data_dir', project_dir / 'data'))
        models_dir = to_path(kwargs_get('models_dir', project_dir / 'models'))

        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        self.project_dir = str(to_path(project_dir))
        self.raw_data_json_path = str(to_path(raw_data_json_path))
        self.data_dir = str(to_path(data_dir))
        self.models_dir = str(to_path(models_dir))
        self.language_masker_path = kwargs_get('language_masker_path', data_dir / 'language_masker.json')
        self.training_dataset_path = kwargs_get('training_dataset_path', data_dir / 'training_dataset')
        self.augmentations_dataset_path = kwargs_get('augmentations_dataset_path', data_dir / 'augmentations_dataset')
        self.trained_generator_ckpt = kwargs_get('trained_generator_ckpt', None)
        self.trained_generator_classifier_ckpt = kwargs_get('trained_generator_classifier_ckpt', None)

    @staticmethod
    def func_varnames(_function):
        return tuple(signature(_function).parameters)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]):
        configs = cls(**load_json(json_path))
        return configs

    def to_json(self, json_path: Union[str, Path]):
        configs = self.__dict__
        init_keys = self.func_varnames(self.__init__)
        configs = {k: v for k, v in configs.items() if k in init_keys and k not in ('self', 'args', 'kwargs')}
        save_json(configs, json_path)

    @property
    def tokenizer(self):
        return self.t5_model_name

    @property
    def reload_dataloaders_every_epoch(self):
        if self.refresh_train_dataset_every_n_epochs is None:
            return False
        return self.refresh_train_dataset_every_n_epochs > 0

    @property
    def generate_kwargs(self):
        generate_keys = ['num_beams', 'num_beam_groups', 'repetition_penalty', 'length_penalty', 'diversity_penalty',
                         'early_stopping', 'num_return_sequences', 'do_sample', 'temperature', 'top_k', 'top_p']
        return {k: getattr(self, k) for k in generate_keys}

    def copy(self) -> "Configs":
        return copy(self)

    def varnames_to_kwargs(self, varnames, ignore_none: bool = False,
                           *additional_kwargs, **kwargs_to_update):
        kwargs = {k: getattr(self, k) for k in varnames if hasattr(self, k)}
        if ignore_none:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
        for additional in additional_kwargs:
            if additional not in kwargs and hasattr(self, additional):
                kwargs[additional] = getattr(self, additional)
        kwargs_to_update = {k: v for k, v in kwargs_to_update.items() if k in varnames or k in additional_kwargs}
        kwargs.update(kwargs_to_update)
        return kwargs

    def kwargs_for_class(self, _class, ignore_none: bool = False,
                         *additional_kwargs, **kwargs_to_update):
        varnames = [k for k in self.func_varnames(_class.__init__)
                    if k not in ('self', 'args', 'kwargs')]
        return self.varnames_to_kwargs(varnames, ignore_none, *additional_kwargs, **kwargs_to_update)

    def kwargs_for_function(self, _function, *additional_kwargs, **kwargs_to_update):
        varnames = [k for k in self.func_varnames(_function) if k not in ('args', 'kwargs')]
        return self.varnames_to_kwargs(varnames, *additional_kwargs, **kwargs_to_update)

    def update_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def copy_and_update_kwargs(self, **kwargs) -> "Configs":
        configs = self.copy()
        configs.update_kwargs(**kwargs)
        return configs

    def to_fast_dev_mode(self) -> "Configs":
        return self.copy_and_update_kwargs(generator_epochs=1,
                                           generator_classifier_epochs=1,
                                           num_workers=1,
                                           fast_dev=True,
                                           fast_dev_run=False)
