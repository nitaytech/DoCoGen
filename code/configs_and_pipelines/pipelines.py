from pathlib import Path
from typing import Union, List, Dict, Any
from transformers import T5TokenizerFast
import pandas as pd
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from datasets import Dataset
from modeling.generation.classifier import LightningClassifier
from modeling.generation.controllable_model import LightningControllableT5, ControllableT5Configs
from modeling.generation.trainer import T5Trainer
from project_utils.functions import (to_path, calculate_total_training_steps,
                                     create_experiment_name_and_dir, round_power_two)
from project_utils.constants import BATCH_SIZE, UNK
from data_loading.concepts_dataset import ConceptsDataset, ConceptsDataModule

from data_loading.dataset_utils import filter_dataset
from modeling.masking.text_processor import TextProcessor
from modeling.masking.masker import LanguageMasker
from configs_and_pipelines.configs_manager import Configs
from project_utils.functions import load_json


def preprocess_datasets(dataset: Union[Dataset, Dict[str, List]],
                        tokenizer: Union[str, T5TokenizerFast] = 't5-base',
                        max_length: int = 96) -> Dataset:
    def _is_unknown(x: Any) -> bool:
        if x is None:
            return True
        if isinstance(x, str):
            if x.lower() in ['none', 'unknown', UNK.lower(), 'null']:
                return True
        return False
    processor = TextProcessor(tokenizer, max_seq_len=max_length)
    if isinstance(dataset, Dataset):
        dataset = {k: dataset[k] for k in dataset.column_names}
    if 'example_id' not in dataset:
        dataset['example_id'] = [f'{split}_{i}' for i, split in enumerate(dataset['split'])]
    for k in dataset:  # all columns in dataset should be strings
        if k in ['example_id', 'split', 'text']:
            continue
        dataset[k] = [UNK if _is_unknown(x) else str(x) for x in dataset[k]]
    dataset['text'] = processor.preprocess_text(dataset['text'], lower=False, stem=False,
                                                space_between_punctuations=True, clean_spaces=True,
                                                tokenize_decode=True, max_length=max_length)
    tokenized_data = processor.tokenize_texts(dataset['text'], padding='max_length', truncation=True,
                                              max_length=max_length, return_tensors="np",
                                              return_attention_mask=True)
    dataset['input_ids'] = tokenized_data['input_ids'].tolist()
    dataset['attention_mask'] = tokenized_data['attention_mask'].tolist()
    dataset = Dataset.from_dict(dataset)
    dataset.cleanup_cache_files()
    return dataset


def language_masker_runner(dataset: Dataset,
                           configs: Configs):
    masker_kwargs = configs.kwargs_for_class(LanguageMasker)
    concept_columns = ConceptsDataset.extract_concept_columns(dataset)
    concepts_values = {c: sorted(list(set(dataset[c]))) for c in concept_columns}
    language_masker = LanguageMasker(**masker_kwargs)
    # adding other concepts and their values from the train and validation datasets to concepts_mapping
    for concept, values in concepts_values.items():
        for value in values:
            language_masker.add_concept_value(concept, value)
    language_masker.add_dataset(dataset,
                                text_column='text',
                                concept_columns=concept_columns,
                                min_occurrences=configs.min_occurrences)
    language_masker.add_orientations_top_words(n_orientations=configs.n_orientations,
                                               with_value_name=True,
                                               top_occurrences_threshold=configs.top_occurrences_threshold,
                                               concepts=configs.concepts_to_control)
    language_masker.update_cached_probabilities(smoothing=configs.smoothing)
    language_masker.update_cached_masked_words(using_cached=True)
    language_masker.to_json(configs.language_masker_path)
    return language_masker


def build_datasets(configs: Configs, *,
                   fit_language_masker: bool = True,
                   fast_dev: bool = True) -> Configs:
    if fast_dev:
        configs = configs.to_fast_dev_mode()
    json_dataset_path = configs.raw_data_json_path
    tokenizer = configs.tokenizer
    max_seq_len = configs.max_seq_len
    concept_to_control = configs.concept_to_control
    values_to_control = configs.values_to_control
    splits_for_training = configs.splits_for_training
    splits_for_augmentations = configs.splits_for_augmentations
    training_dataset_path = configs.training_dataset_path
    augmentations_dataset_path = configs.augmentations_dataset_path
    dataset = load_json(json_dataset_path)
    dataset = preprocess_datasets(dataset, tokenizer, max_seq_len)
    if configs.fast_dev:
        kvs = [{'split': splits_for_training, 'n_samples': 100},
               {'split': splits_for_augmentations, 'n_samples': 100}]
        dataset = filter_dataset(dataset, key_values_to_keep=kvs)
    training_kvs = [{'split': splits_for_training, concept_to_control: values_to_control}]
    training_dataset = filter_dataset(dataset, key_values_to_keep=training_kvs)
    if fit_language_masker:
        language_masker = language_masker_runner(dataset=training_dataset, configs=configs)
    else:
        language_masker = LanguageMasker.from_json(configs.language_masker_path)
    augmentations_kvs = [{'split': splits_for_augmentations}]
    augmentations_dataset = filter_dataset(dataset, key_values_to_keep=augmentations_kvs)
    # splitting the training dataset to train, validation and test
    training_dataset = {c: training_dataset[c] for c in training_dataset.column_names}
    train_size = len(training_dataset['split'])
    val_size = min(max(round_power_two(int(train_size ** 0.5)), configs.batch_size * 2), int(0.25 * train_size))
    training_indices = list(range(train_size))
    val_indices = language_masker.rnp.choice(training_indices, size=val_size, replace=False)
    for idx in val_indices:
        training_dataset['split'][idx] = 'validation'
    test_indices = language_masker.rnp.choice([idx for idx in training_indices if idx not in val_indices],
                                              size=val_size, replace=False)
    for idx in test_indices:
        training_dataset['split'][idx] = 'test'
    for idx in set(training_indices) - set(val_indices) - set(test_indices):
        training_dataset['split'][idx] = 'train'
    training_dataset = Dataset.from_dict(training_dataset)
    dataset_kwargs = configs.kwargs_for_class(ConceptsDataset, language_masker=language_masker)
    training_dataset = ConceptsDataset(dataset=training_dataset, **dataset_kwargs)
    training_dataset.refresh()
    training_dataset.to_json(training_dataset_path, new_language_masker_path=False)

    augmentations_dataset = ConceptsDataset(dataset=augmentations_dataset, **dataset_kwargs)
    augmentations_dataset = augmentations_dataset.create_new_instance_for_augmentations(
        save_new_instance_to_dir=augmentations_dataset_path,
        sampling_seed=None,
        columns=augmentations_dataset.dataset.column_names,
        concepts_values_for_augmentations={concept_to_control:
                                           language_masker.concepts_mapping.concept_values(concept_to_control)},
        n_augmentations_per_example='all',
        include_unknown_orientation=False,
    )
    augmentations_dataset.to_json(augmentations_dataset_path)
    return configs


def data_module_kwargs_for_classification(configs):
    kwargs = dict(
        batch_size=configs.generator_classifier_batch_size,
        start_probabilities=(('classifier', 1.0),),
        end_probabilities=(('classifier', 1.0),),
        n_steps=0,
        evaluation_step='classifier',
        refresh_train_dataset_every_n_epochs=None,
        refresh_mlm=False,
    )
    return kwargs


def kwargs_for_classifier(configs: Configs):
    _, output_dir = create_experiment_name_and_dir(output_dir=configs.models_dir,
                                                   experiment_name='classifier',
                                                   add_datetime=False)
    kwargs = dict(
        output_dir=output_dir,
        evaluation_step='classifier',
        batch_size=configs.generator_classifier_batch_size,
        start_probabilities=(('classifier', 1.0),),
        end_probabilities=(('classifier', 1.0),),
        n_steps=0,
        refresh_train_dataset_every_n_epochs=None,
        refresh_mlm=False,
        ignore_unknown=False,
        concepts_to_predict=configs.concepts_to_control,
        max_epochs=configs.generator_classifier_epochs,
        monitor='validation_mean_predict_acc',
        mode='max',
    )
    return kwargs


def kwargs_for_generator(configs: Configs):
    _, output_dir = create_experiment_name_and_dir(output_dir=configs.models_dir,
                                                   experiment_name='generator',
                                                   add_datetime=False)
    kwargs = dict(
        output_dir=output_dir,
        evaluation_step='generation',
        start_probabilities=(('mlm', 1.0), ('classifier', 0.0)),
        end_probabilities=(('mlm', 1.0), ('classifier', 0.0)),
        n_steps=0,
        max_epochs=configs.generator_epochs,
        monitor='validation_mean_generated_predict_acc',
        mode='max',
    )
    return kwargs


def prepare_data_module(experiment_type: str,
                        concepts_dataset_path: Union[str, Path],
                        configs: Configs) -> ConceptsDataModule:
    assert experiment_type in ['generator', 'classifier', 'generation']
    if experiment_type == 'classifier':
        data_module_kwargs = configs.kwargs_for_class(ConceptsDataModule,
                                                      **data_module_kwargs_for_classification(configs))
    else:
        data_module_kwargs = configs.kwargs_for_class(ConceptsDataModule)
    seed = data_module_kwargs.get('seed', None)
    concepts_dataset = ConceptsDataset.from_json(concepts_dataset_path)
    train_settings = [{'split': ['train']}]
    val_settings = [{'split': ['validation']}]
    test_settings = [{'split': ['test']}]
    if configs.fast_dev:
        batch_size = data_module_kwargs.get('batch_size', BATCH_SIZE)
        train_settings[0]['n_samples'] = batch_size
        val_settings[0]['n_samples'] = batch_size
        test_settings[0]['n_samples'] = batch_size
    concepts_data_module = ConceptsDataModule.from_concepts_dataset(concepts_dataset=concepts_dataset,
                                                                    train_settings=train_settings,
                                                                    val_settings=val_settings,
                                                                    test_settings=test_settings,
                                                                    sampling_seed=seed,
                                                                    **data_module_kwargs)
    return concepts_data_module


def prepare_model(experiment_type: str,
                  configs: Configs,
                  trained_generator_ckpt: Union[str, Path, None] = None,
                  trained_generator_classifier_ckpt: Union[str, Path, None] = None) -> Union[LightningControllableT5, LightningClassifier]:
    assert experiment_type in ['generator', 'classifier', 'generation']
    if experiment_type in ['generator', 'generation']:
        model_configs = ControllableT5Configs(**configs.kwargs_for_class(ControllableT5Configs, ignore_none=True))
        model_kwargs = configs.kwargs_for_class(LightningControllableT5, ignore_none=True)
        model_kwargs['configs'] = model_configs
        if trained_generator_ckpt is not None:
            model = LightningControllableT5.load_from_checkpoint_workaround(trained_generator_ckpt, **model_kwargs)
        else:
            model = LightningControllableT5(**model_kwargs)
        if trained_generator_classifier_ckpt is not None:
            trained_classifier = LightningClassifier.load_from_checkpoint_workaround(trained_generator_classifier_ckpt)
            model.classifier = trained_classifier.classifier
    else:  # experiment_type in ['classifier']
        model_kwargs = configs.kwargs_for_class(LightningClassifier, ignore_none=True)
        if trained_generator_classifier_ckpt is not None:
            model = LightningClassifier.load_from_checkpoint_workaround(trained_generator_classifier_ckpt, **model_kwargs)
        else:
            model = LightningClassifier(**model_kwargs)
    return model


def prepare_configs_for_experiment(configs: Configs,
                                   output_dir: str,
                                   concepts_dataset: ConceptsDataset) -> Configs:
    train_len = len(concepts_dataset)
    calculate_steps_kwargs = configs.kwargs_for_function(calculate_total_training_steps,
                                                         'train_len', train_len=train_len)
    training_steps = calculate_total_training_steps(**calculate_steps_kwargs)
    concepts_mapping = concepts_dataset.concepts_mapping
    if concepts_dataset.language_masker_path is not None:
        language_masker = concepts_dataset.language_masker_path
    else:
        language_masker = concepts_dataset.language_masker
    return configs.copy_and_update_kwargs(output_dir=output_dir,
                                          n_steps=training_steps,
                                          training_steps=training_steps,
                                          concepts_mapping=concepts_mapping,
                                          language_masker=language_masker)


def prepare_experiment(experiment_type: str,
                       output_dir: str,
                       concepts_dataset_path: Union[str, Path],
                       configs: Configs,
                       trained_generator_ckpt: Union[str, Path, None] = None,
                       trained_generator_classifier_ckpt: Union[str, Path, None] = None) -> Dict[str, Any]:
    assert experiment_type in ['generator', 'classifier', 'generation']
    concepts_data_module = prepare_data_module(experiment_type=experiment_type,
                                               concepts_dataset_path=concepts_dataset_path,
                                               configs=configs)
    configs = prepare_configs_for_experiment(configs, output_dir, concepts_data_module.train_dataset)
    model = prepare_model(experiment_type=experiment_type,
                          configs=configs,
                          trained_generator_ckpt=trained_generator_ckpt,
                          trained_generator_classifier_ckpt=trained_generator_classifier_ckpt)
    trainer_kwargs = configs.kwargs_for_class(T5Trainer)
    trainer = T5Trainer(**trainer_kwargs)
    return {'output_dir': output_dir,
            'data_module': concepts_data_module,
            'model': model, 'trainer': trainer,
            'new_model_ckpt_path': to_path(output_dir) / f"{configs.monitor}_checkpoint.ckpt"}


def run_training_experiment(model: LightningModule,
                            data_module: LightningDataModule,
                            trainer: Trainer):
    trainer.fit(model, datamodule=data_module)
    return trainer.test(model, datamodule=data_module)


def train_models(configs: Configs,
                 train_generator_classifier: bool = True,
                 train_generator: bool = True,
                 fast_dev: bool = False) -> Configs:
    if fast_dev:
        runner_configs = configs.to_fast_dev_mode()
    else:
        runner_configs = configs.copy()
    if train_generator_classifier:
        generator_classifier_configs = runner_configs.copy_and_update_kwargs(**kwargs_for_classifier(runner_configs))
        experiment_setup = prepare_experiment(experiment_type='classifier',
                                              output_dir=generator_classifier_configs.output_dir,
                                              concepts_dataset_path=generator_classifier_configs.training_dataset_path,
                                              configs=generator_classifier_configs,
                                              trained_generator_classifier_ckpt=None,
                                              trained_generator_ckpt=None)
        data_module = experiment_setup.get('data_module')
        model = experiment_setup.get('model')
        trainer = experiment_setup.get('trainer')
        run_training_experiment(model, data_module, trainer)
        runner_configs.trained_generator_classifier_ckpt = str(experiment_setup.get('new_model_ckpt_path'))
        configs.trained_generator_classifier_ckpt = str(experiment_setup.get('new_model_ckpt_path'))
    if train_generator:
        generator_configs = runner_configs.copy_and_update_kwargs(**kwargs_for_generator(runner_configs))
        experiment_setup = prepare_experiment(experiment_type='generator',
                                              output_dir=generator_configs.output_dir,
                                              concepts_dataset_path=generator_configs.training_dataset_path,
                                              configs=generator_configs,
                                              trained_generator_classifier_ckpt=runner_configs.trained_generator_classifier_ckpt,
                                              trained_generator_ckpt=None)
        data_module = experiment_setup.get('data_module')
        model: LightningControllableT5 = experiment_setup.get('model')
        trainer = experiment_setup.get('trainer')
        run_training_experiment(model, data_module, trainer)
        configs.trained_generator_ckpt = str(experiment_setup.get('new_model_ckpt_path'))
        model.model.save_model(to_path(experiment_setup.get('output_dir')) / 'model')
    return configs


def generate(configs: Configs,
             splits: List[str] = ('train', 'validation', 'test'),
             fast_dev: bool = False):
    if fast_dev:
        runner_configs = configs.to_fast_dev_mode()
    else:
        runner_configs = configs.copy()
    generator_configs = runner_configs.copy_and_update_kwargs(**kwargs_for_generator(runner_configs))
    generator_configs.output_dir = to_path(str(generator_configs.output_dir.replace('generator', 'generation')))
    experiment_setup = prepare_experiment(experiment_type='generation',
                                          output_dir=generator_configs.output_dir,
                                          concepts_dataset_path=generator_configs.augmentations_dataset_path,
                                          configs=generator_configs,
                                          trained_generator_classifier_ckpt=generator_configs.trained_generator_classifier_ckpt,
                                          trained_generator_ckpt=generator_configs.trained_generator_ckpt,)
    output_dir = to_path(experiment_setup['output_dir'])
    data_module: ConceptsDataModule = experiment_setup.get('data_module')
    model = experiment_setup.get('model')
    trainer = experiment_setup.get('trainer')
    datasets = {'train': data_module.train_dataset,
                'validation': data_module.val_dataset,
                'test': data_module.test_dataset}
    for split in splits:
        dataset = datasets.get(split)
        if dataset is None:
            continue
        data_module.test_dataset = dataset
        trainer.test(model, datamodule=data_module)
        # filter and use only relevant columns
        outputs_dict = load_json(output_dir / f"test_outputs.json")
        pd.DataFrame(outputs_dict).to_csv(output_dir / f"{split}_generations.csv", index=False)
