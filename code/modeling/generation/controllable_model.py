from argparse import Namespace
from pathlib import Path
from typing import List, Any, Dict, Union, Tuple

import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, LongTensor
from torch.nn import Module
from transformers import (BatchEncoding, AdamW, get_linear_schedule_with_warmup,
                          T5ForConditionalGeneration)
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.file_utils import ModelOutput
from modeling.generation.modules import OrientationEmbeddingDict, OrientationPossibleIdsDict
from modeling.generation.huggingface_generation_utils import GenerationMixin
from modeling.masking.masker import LanguageMasker
from modeling.generation.classifier import EncoderClassifier
from modeling.masking.text_processor import TextProcessor
from project_utils.constants import UNK, LOSS_IGNORE_ID, SEP, MAX_SEQ_LEN, T5_MODEL_NAME
from project_utils.functions import (to_path, concat_dicts, subset_dict, load_json, save_json, repeat_lists_in_dict)


class ControllableT5Configs:
    def __init__(self,
                 t5_model_name: str = T5_MODEL_NAME,
                 max_seq_len: int = MAX_SEQ_LEN,
                 language_masker: Union[str, Path, LanguageMasker] = None,
                 concepts_to_control: List[str] = None,
                 concepts_orientations: Dict[str, List[str]] = None,
                 possible_words_for_generation: Union[bool, Dict[str, Dict[str, List[str]]]] = None,
                 unknown_orientation: Union[str, None] = UNK):
        """
        A configs class for the controllable T5 generator.
        :param t5_model_name:  a string [default: 't5-base'], can be: 't5-small', 't5-base', 't5-large'.
        :param max_seq_len: an int [default: 96]. The maximal size of the generated texts.
        :param language_masker: a path-like or a LanguageMasker object (used to mask the texts).
        :param concepts_to_control: list of strings of the concepts you wish to control.
        :param concepts_orientations: a dict where the keys are controllable concepts, and the values are lists
        of possible orientations, e.g. {'domain': ['dvd', 'plot', 'airline', 'crew']}
        :param possible_words_for_generation: if given a bool - then it uses the language_masker to build the dict.
        If a dict is given, the keys should be the controllable concepts, the values are also dicts with orientations
        as keys and a list of possible words which should be used during the generation.
        :param unknown_orientation: a string that indicates what is the unknown orientation.
        """
        assert t5_model_name in ['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'],\
            "`t5_model_name` should be: 't5-small', 't5-base', 't5-large', 't5-3b' or 't5-11b'."
        self.t5_model_name = t5_model_name
        assert isinstance(max_seq_len, int), "`max_seq_len` must be an int."
        self.max_seq_len = max_seq_len
        # read language_masker
        if isinstance(language_masker, (str, Path)):
            self.language_masker_path = language_masker
            self.language_masker = LanguageMasker.from_json(language_masker)
        else:
            self.language_masker_path = None
        assert isinstance(self.language_masker, LanguageMasker), "`language_masker` should be LanguageMasker."
        # prepare concepts parameters
        if concepts_to_control is not None:
            assert set(concepts_to_control).issubset(set(self.language_masker.concepts)), \
                "`language_masker` should contain all concepts in `concepts_to_control`."
            concepts_to_control = [c for c in concepts_to_control if c in self.language_masker.concepts]
        else:
            concepts_to_control = self.language_masker.concepts
        self.concepts_to_control = concepts_to_control
        if concepts_orientations is None:
            concepts_orientations = {c: o for c, o in self.language_masker.concepts_ordered_orientations.items()}
        self.concepts_orientations = {c: o for c, o in concepts_orientations.items() if c in concepts_to_control}
        assert set(self.concepts_orientations).issubset(set(self.concepts_to_control)), \
            "There are concepts in `concepts_to_control` which are not in `concepts_orientations`."
        if possible_words_for_generation is None or not possible_words_for_generation:
            if len(self.language_masker.masked_words) == 0:
                self.language_masker.update_cached_masked_words()
            opwfg = {}
            for concept, orientations in self.concepts_orientations.items():
                opwfg[concept] = {o: None for o in orientations}
                if possible_words_for_generation is None:
                    for value, v_orientations in self.language_masker.concept_orientations(concept).items():
                        possible_words = self.language_masker.masked_words[concept][value]
                        for v_orientation in v_orientations:
                            if v_orientation in orientations:
                                opwfg[concept][v_orientation] = possible_words
            possible_words_for_generation = opwfg
        self.possible_words_for_generation = {c: o for c, o in possible_words_for_generation.items()
                                              if c in self.concepts_to_control}
        self.unknown_orientation = unknown_orientation

    def to_json(self, json_path: Union[str, Path]):
        json_path = to_path(json_path)
        if self.language_masker_path is None:
            parts = list(json_path.parts)
            parts[-1] = parts[-1].replace(json_path.stem, f'{json_path.stem}_language_masker')
            self.language_masker_path = str(Path(*parts))
            self.language_masker.to_json(self.language_masker_path)
        configs_dict = {'t5_model_name': self.t5_model_name,
                        'max_seq_len': self.max_seq_len,
                        'language_masker': self.language_masker_path,
                        'concepts_to_control': self.concepts_to_control,
                        'concepts_orientations': self.concepts_orientations,
                        'possible_words_for_generation': self.possible_words_for_generation,
                        'unknown_orientation': self.unknown_orientation}
        save_json(configs_dict, json_path)

    @classmethod
    def from_json(cls, json_path: Union[str, Path], **new_kwargs):
        configs_dict = load_json(json_path)
        configs_dict.update(new_kwargs)
        return cls(**configs_dict)


class ControllableT5(Module, ModuleUtilsMixin, GenerationMixin):
    def __init__(self,
                 configs: Union[ControllableT5Configs, str, Path] = None,
                 **configs_new_kwargs):
        """

        :param configs: a path-like or a ControllableT5Configs object.
        :param configs_new_kwargs: overloading the configs with these new key-word arguments.
        """
        super().__init__()
        if configs is None:
            configs = ControllableT5Configs(**configs_new_kwargs)
        elif isinstance(configs, (str, Path)):
            configs = ControllableT5Configs.from_json(configs, **configs_new_kwargs)
        assert isinstance(configs, ControllableT5Configs), "`configs` should be ControllableT5Configs"
        self.configs = configs
        self.model = T5ForConditionalGeneration.from_pretrained(self.configs.t5_model_name)
        processor = TextProcessor(tokenizer=self.configs.t5_model_name,
                                  max_seq_len=self.configs.max_seq_len,
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)
        self.language_masker.processor = processor
        self.concepts_embeddings = OrientationEmbeddingDict.from_concepts_orientations(
            self.configs.concepts_orientations, lm_embedding_module=self.model.shared, lm_tokenizer=self.tokenizer,
            embedding_dim=self.model.model_dim, unknown_orientation=self.configs.unknown_orientation)
        self.concepts_embeddings.to(self.model.device)
        self.possible_ids = OrientationPossibleIdsDict(
            self.configs.possible_words_for_generation, lm_tokenizer=self.tokenizer,
            unknown_orientation=self.configs.unknown_orientation)

    @property
    def language_masker(self):
        return self.configs.language_masker

    @property
    def processor(self):
        return self.language_masker.processor

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def concepts_to_control(self):
        return self.configs.concepts_to_control

    @property
    def orientations(self):
        return {c: {k: list(o) for k, o in v.items()}
                for c, v in self.language_masker.orientations.items() if c in self.concepts_to_control}

    # ----- properties and methods used for GenerationMixin -----

    @property
    def config(self):
        return self.model.config

    def get_encoder(self):
        return self.model.get_encoder()

    def _reorder_cache(self, past, beam_idx):
        return self.model._reorder_cache(past=past, beam_idx=beam_idx)

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None,
                                      use_cache=None, encoder_outputs=None,
                                      **kwargs):
        return self.model.prepare_inputs_for_generation(input_ids=input_ids, past=past, attention_mask=attention_mask,
                                                        use_cache=use_cache, encoder_outputs=encoder_outputs, **kwargs)

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: LongTensor, model_kwargs) -> Dict[str, Any]:

        # retrieve encoder hidden states
        encoder_kwargs = {
            argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
        }
        if 'orientations' in model_kwargs:
            model_kwargs.pop('orientations')
        # encoder_kwargs contains attention mask
        with torch.no_grad():
            model_kwargs["encoder_outputs"], model_kwargs['attention_mask'] = self.add_orientations_and_encode(
                input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs

    # ----- generation methods -----

    def tokenize_texts(self, batch_texts,
                       return_tensors: str = 'pt'):
        return self.processor.tokenize_texts(batch_texts, padding='max_length', truncation=True,
                                             max_length=self.configs.max_seq_len,
                                             return_tensors=return_tensors, return_attention_mask=True)

    def decode_texts(self, batch_sequences,
                     skip_special_tokens: bool = None,
                     remove_pad_and_eos: bool = True,
                     clean_up_tokenization_spaces: bool = None):
        decoded = self.processor.decode_texts(batch_sequences,
                                              skip_special_tokens=skip_special_tokens,
                                              clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        if remove_pad_and_eos:
            decoded = [s.replace(self.tokenizer.pad_token, '').replace(self.tokenizer.eos_token, '') for s in decoded]
            decoded = [s.replace('<extra_id_', ' <extra_id_') for s in decoded]
            decoded = [s[1:] if s.startswith(' <extra_id_') else s for s in decoded]
        decoded = [self.processor.clean_spaces(t) for t in decoded]
        return decoded

    def generate_ids(self, input_ids: LongTensor,
                     attention_mask: LongTensor,
                     orientations: Dict[str, List[str]],
                     use_oriented_processor: bool = True,
                     **kwargs) -> LongTensor:
        if use_oriented_processor:
            oriented_processor = self.possible_ids.prepare_oriented_processor(input_ids, orientations)
        else:
            oriented_processor = None
        return self.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             orientations=orientations,
                             additional_logits_processors=oriented_processor,
                             max_length=kwargs.pop('max_length', self.configs.max_seq_len),
                             num_beams=kwargs.pop('num_beams', 4),
                             num_beam_groups=kwargs.pop('num_beam_groups', 4),
                             repetition_penalty=kwargs.pop('repetition_penalty', 2.0),
                             length_penalty=kwargs.pop('length_penalty', 1.5),
                             early_stopping=kwargs.pop('early_stopping', True),
                             diversity_penalty=kwargs.pop('diversity_penalty', 0.05),
                             num_return_sequences=kwargs.pop('num_return_sequences', 1),
                             do_sample=kwargs.pop('do_sample', False),
                             temperature=kwargs.pop('temperature', 1.0),
                             top_k=kwargs.pop('top_k', 40),
                             top_p=kwargs.pop('top_p', 1.0),
                             **kwargs)

    # ----- generation with masking methods -----

    def mask_texts(self, batch_texts: List[str],
                   concepts_values: Dict[str, List[str]],
                   orientations: Dict[str, List[str]],
                   **masking_kwargs) -> Dict[str, Any]:
        """
        Masks the `batch_texts`.
        :param batch_texts: a list of textual examples.
        :param concepts_values: a dict where the keys are the controllable concepts and the values are lists of
            their values. The i-th entry corresponds to the value of the i-th example in the batch. These values
            are used for calculating the masking scores of the term.
        :param orientations: a dict where the keys are the controllable concepts and the values are lists of
            orientations. The i-th entry corresponds to the orientation of the i-th example in the batch.
        :param masking_kwargs: possible masking kwargs, if not provided using the the values in the configurations.
            * smoothing: a string, or a list or tuple of ints [default: (1, 5, 7)]. The smoothing hyperparameter $a$
                which is used when estimating the probability of the controlled value $D$ given the term $w$:
                $P(D|w)=#_{w|D}+a/n_D$. If a string is given it should be 'n', in that case the smoothing hyperparameter
                will be eual to the number of words in the term. If a list or tuple of ints is given, it should be as
                 the size of `n_grams`, and entry i corresponds to the smoothing hyperparameter used for an n-gram
                 of size i. For example if `smoothing_hyperparameter =(1, 3, 7)` then 1, 3 and 7 are the smoothing
                 hyperparameters of uni-grams, bi-grams and tri-grams respectively.
            * masked_output: a bool. The generator (T5) can be trained to generate a only the
                masked spans) or to whole output. For example, given the input 'I <extra_id_0> you',
                 the output can be either '<extra_id_0> love <extra_id_1>' or 'I love you'.
            * threshold: a float. Terms (n-grams) with a masking score above `threshold` are masked.
                The masking score of a term with an origin value $D$ and a destination value $D'$ is
                $(1-H(D|w)/logN)(P(D|w)-P(D'|w)$.
            * top_n:. A fraction of words with the highest masking score which are masked
                (including terms with a masking score above the threshold).
            * noise: . A fraction of additional randomly masked words.
            * n_grams: an int (maximum n-gram size) or a list of ints (possible n-grams,
                e.g. [1, 3] for uni-grams and tri-grams).
            * using_cached: a bool, if True using the chached probabilities of language_masker (faster compute).
            * return_probabilities: if True returns a dict where the keys are the controllable concepts and the values
                are lists with probabilities of each term.
        :return: a dict with the following key-values:
            * 'mlm_input_ids' - LongTensor of tokenized masked texts.
            * 'mlm_attention_mask' - LongTensor of the attention mask of the tokenized masked texts.
            * 'mlm_output_input_ids' - If `masked_output` == True, will include LongTensor of the
             tokenized masked terms. Otherwise, will include the tokenized `batch_texts`.
            * 'masked_output_text' - If `masked_output` == True, will include the list of the masked terms.
                Otherwise will include `batch_texts`.
            * 'probabilities' (only if `return_probabilities` == True):
        """
        concepts_values, orientations, new_concepts_values = self._fix_inputs(len(batch_texts), concepts_values,
                                                                              orientations, new_concepts_values=None)

        noise = masking_kwargs.pop('noise', 0.0)
        device = masking_kwargs.pop('device', self.device)
        return_tensors = masking_kwargs.pop('return_tensors', 'pt')
        return self.language_masker.words_concepts_modeling_mask(batch_texts, concepts_values, new_concepts_values,
                                                                 noise=noise, device=device,
                                                                 return_tensors=return_tensors, **masking_kwargs)

    def mask_and_generate_ids(self, batch_texts: List[str],
                              concepts_values: Dict[str, List[str]],
                              orientations: Dict[str, List[str]],
                              use_oriented_processor: bool = True,
                              *, return_masked_texts: bool = False,
                              **kwargs) -> Union[LongTensor, Tuple[LongTensor, List[str]]]:
        """
        Masks the `batch_texts` according to the `orientations` and then generates token ids according to them.
        :param batch_texts: a list of textual examples.
        :param concepts_values: a dict where the keys are the controllable concepts and the values are lists of
            their values. The i-th entry corresponds to the value of the i-th example in the batch. These values
            are used for calculating the masking scores of the term.
        :param orientations: a dict where the keys are the controllable concepts and the values are lists of
            orientations. The i-th entry corresponds to the orientation of the i-th example in the batch.
        :param use_oriented_processor: a bool, if True, uses the oriented processor which force the decoding method
            to generate only the possible token ids as given in the configs.
        :param return_masked_texts: a bool, if True returns also the masked texts.
        :param kwargs: may include masking kwargs (see self.mask_texts() documentation) or generation kwargs
            (see https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate)
        :return: LongTensor of the generated token ids.
            If `return_masked_texts`, then returns a tuple where the first arg is the generated tokens and the
             second arg is a list of masked texts.
        """
        concepts_values, orientations, _ = self._fix_inputs(len(batch_texts), concepts_values,
                                                            orientations, new_concepts_values=None)
        mkeys = ['masked_output', 'threshold', 'top_n', 'noise', 'mask_scores_pooling',
                 'smoothing', 'n_grams', 'using_cached']
        masking_kwargs = {k: kwargs.pop(k) for k in mkeys if k in kwargs}
        masking_kwargs.update({'return_tensors': 'pt', 'device': self.device, 'return_probabilities': False})
        masked_batch = self.mask_texts(batch_texts, concepts_values, orientations, **masking_kwargs)
        generated_ids = self.generate_ids(masked_batch['mlm_input_ids'], masked_batch['mlm_attention_mask'],
                                          orientations, use_oriented_processor, **kwargs)
        if return_masked_texts:
            return generated_ids, masked_batch['masked_text']
        else:
            return generated_ids

    def mask_and_generate_texts(self, batch_texts: List[str],
                                concepts_values: Dict[str, List[str]],
                                orientations: Dict[str, List[str]],
                                use_oriented_processor: bool = True,
                                *, return_masked_texts: bool = False,
                                **kwargs) -> Union[List[str], Tuple[List[str], List[str]]]:
        """
        Masks the `batch_texts` according to the `orientations` and then generates counterfactuals according to them.
        parameters: see self.mask_and_generate_ids() documentation.
        :return: list of generated counterfactuals.
            If `return_masked_texts`, then returns a tuple where the first arg is the generated texts and the
             second arg is a list of masked texts.
        """
        generated_ids, masked_text = self.mask_and_generate_ids(batch_texts, concepts_values,
                                                                orientations, use_oriented_processor,
                                                                return_masked_texts=True, **kwargs)
        if self.language_masker.masked_output:
            generated_text = self.decode_texts(generated_ids, skip_special_tokens=False, remove_pad_and_eos=True)
            generated_text = self.language_masker.combine_input_output(masked_text, generated_text)
        else:
            generated_text = self.decode_texts(generated_ids)
        generated_text = [self.processor.clean_spaces(t) for t in generated_text]
        if return_masked_texts:
            return generated_text, masked_text
        else:
            return generated_text

    # ----- forward methods -----

    def _fix_inputs(self,
                    batch_size: int,
                    concepts_values: Dict[str, List[str]] = None,
                    orientations: Dict[str, List[str]] = None,
                    new_concepts_values: Dict[str, List[str]] = None):
        if concepts_values is None:
            concepts_values = {}
        if orientations is None:
            orientations = {}
        if new_concepts_values is None:
            new_concepts_values = {}
        for concept in self.concepts_to_control:
            if concept not in concepts_values:
                concepts_values[concept] = [UNK for _ in range(batch_size)]
            if concept not in orientations:
                orientations[concept] = [UNK for _ in range(batch_size)]
            if concept not in new_concepts_values:
                new_concepts_values[concept] = []
                for cv, o in zip(concepts_values[concept], orientations[concept]):
                    ov = self.language_masker.orientation_to_value(concept, o)
                    new_value = ov if ov != UNK else cv  # if the orientation is UNK, we assume the concept is unchanged
                    new_concepts_values[concept].append(new_value)
        concepts_values = {c: v for c, v in concepts_values.items() if c in self.concepts_to_control}
        orientations = {c: v for c, v in orientations.items() if c in self.concepts_to_control}
        new_concepts_values = {c: v for c, v in new_concepts_values.items() if c in self.concepts_to_control}
        return concepts_values, orientations, new_concepts_values

    def get_orientations_embeddings(self, batch_size: int, orientations: Dict[str, List[str]]) -> Tensor:
        _, orientations, _ = self._fix_inputs(batch_size, orientations=orientations)
        embeddings = self.concepts_embeddings.get_orientations_embeddings(batch_size, orientations)
        return embeddings

    def add_orientations_and_encode(self, input_ids: LongTensor,
                                    attention_mask: LongTensor,
                                    orientations: Dict[str, List[str]],
                                    **encoder_kwargs) -> Tuple[ModelOutput, LongTensor]:
        inputs_embeds = self.model.encoder.embed_tokens(input_ids)
        orientations = self.get_orientations_embeddings(input_ids.shape[0], orientations)
        inputs_embeds = torch.cat((orientations, inputs_embeds), dim=1)
        attention_mask = torch.cat((torch.ones(size=orientations.shape[:2], device=attention_mask.device,
                                               dtype=attention_mask.dtype), attention_mask), dim=1)
        encoder_outputs = self.model.encoder(input_ids=None,
                                             attention_mask=attention_mask,
                                             inputs_embeds=inputs_embeds,
                                             **encoder_kwargs)
        return encoder_outputs, attention_mask

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels: LongTensor = None,
                encoder_outputs=None,
                orientations: Dict[str, List[str]] = None,
                replace_pad_with_ignore_loss: bool = False,
                **kwargs):
        if encoder_outputs is None:
            # need to add the embeddings and then encode
            encoder_outputs, attention_mask = self.add_orientations_and_encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                orientations=orientations,
                head_mask=kwargs.get('head_mask', None),
                output_attentions=kwargs.get('output_attentions', None),
                output_hidden_states=kwargs.get('output_hidden_states', None),
                return_dict=kwargs.get('return_dict', None))
        # need to add ones to attention mask.
        if attention_mask.shape[1] != encoder_outputs['last_hidden_state'].shape[1]:
            ones_to_add = encoder_outputs['last_hidden_state'].shape[1] - attention_mask.shape[1]
            attention_mask = torch.cat((torch.ones(size=(attention_mask.shape[0], ones_to_add),
                                                   device=attention_mask.device, dtype=attention_mask.dtype),
                                        attention_mask), dim=1)
        if replace_pad_with_ignore_loss:
            # Ignore the pad token during loss calculation by replacing the pad_token_id with LOSS_IGNORE_ID.
            labels[labels[:, :] == self.tokenizer.pad_token_id] = LOSS_IGNORE_ID
        return self.model(input_ids=None,
                          attention_mask=attention_mask,
                          encoder_outputs=encoder_outputs,
                          labels=labels,
                          **kwargs)

    def save_model(self, dir_path: Union[str, Path], save_language_masker: bool = True):
        dir_path = to_path(dir_path, make_parent_dir=True)
        configs_path = dir_path / 'configs.json'
        state_dict_path = dir_path / 'state_dict.pt'
        if save_language_masker:
            self.configs.language_masker_path = None  # this way the language_masker will be saved in dir_path
        self.configs.to_json(configs_path)
        torch.save(self.state_dict(), state_dict_path)

    @classmethod
    def load_model(cls, dir_path: Union[str, Path]):
        dir_path = to_path(dir_path)
        configs_path = dir_path / 'configs.json'
        state_dict = torch.load(dir_path / 'state_dict.pt')
        configs = ControllableT5Configs.from_json(configs_path)
        model = cls(configs)
        model.load_state_dict(state_dict)
        return model


class LightningControllableT5(LightningModule):
    def __init__(self,
                 configs: Union[ControllableT5Configs, str, Path],
                 output_dir: str,
                 # model optimizer_args
                 optimizer_weight_decay: float = 1e-5,
                 optimizer_lr: float = 5e-5,
                 optimizer_eps: float = 1e-8,

                 # scheduler args (for optimizer)
                 training_steps: int = None,
                 warmup_steps: int = 0,
                 **kwargs):
        super().__init__()
        if isinstance(configs, (str, Path)):
            configs = ControllableT5Configs.from_json(configs)
        configs_path = str(Path(output_dir) / 'configs.json')
        configs.to_json(configs_path)
        configs = configs_path
        self.model = ControllableT5(configs)
        self.save_hyperparameters('configs', 'output_dir', 'optimizer_weight_decay', 'optimizer_lr', 'optimizer_eps',
                                  'training_steps', 'warmup_steps')
        encoder_model = kwargs.get('encoder_model', EncoderClassifier.init_encoder_model(self.configs.t5_model_name))
        self.classifier = EncoderClassifier(encoder_model=encoder_model,
                                            concepts_mapping=self.configs.language_masker.concepts_mapping,
                                            concepts_to_predict=self.model.concepts_to_control,
                                            ignore_unknown=kwargs.get('ignore_unknown', False),
                                            label_smoothing=kwargs.get('label_smoothing', 0.2))

    @property
    def configs(self):
        return self.model.configs

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def _forward_step(self, input_ids=None,
                      attention_mask=None,
                      labels: LongTensor = None,
                      orientations: Dict[str, List[str]] = None,
                      **kwargs) -> Dict[str, Tensor]:
        outputs = self.forward(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels,
                               orientations=orientations,
                               replace_pad_with_ignore_loss=True,
                               **kwargs)
        return {'loss': outputs['loss']}

    def _predict_concepts_forward_step(self, input_ids: LongTensor,
                                       attention_mask: LongTensor,
                                       concepts_values: Dict[str, List[str]],
                                       ignore_unknown: bool = True,
                                       return_predictions: bool = False,
                                       return_probabilities: bool = False,
                                       with_encoder_grad: bool = True) -> Dict[str, Any]:
        return self.classifier.predict_concepts_forward_step(input_ids=input_ids,
                                                             attention_mask=attention_mask,
                                                             concepts_values=concepts_values,
                                                             ignore_unknown=ignore_unknown,
                                                             return_predictions=return_predictions,
                                                             return_probabilities=return_probabilities,
                                                             with_encoder_grad=with_encoder_grad)

    def _mlm_forward_step(self, mlm_input_ids: LongTensor,
                          mlm_attention_mask: LongTensor,
                          mlm_output_input_ids: LongTensor,
                          orientations: Dict[str, List[str]]) -> Dict[str, Any]:
        loss_dict = self._forward_step(input_ids=mlm_input_ids,
                                       attention_mask=mlm_attention_mask,
                                       labels=mlm_output_input_ids,
                                       orientations=orientations)
        loss_dict['mlm_loss'] = loss_dict['loss']
        return loss_dict

    def _batch_forward_step(self, batch) -> Dict[str, Any]:
        loss_dict = self._init_loss_dict(eval_mode=False, batch_size=len(batch['example_id']), return_keys=False)
        step_type = batch['step_type']
        assert step_type in ['classifier', 'mlm'], "`step_type` must be 'classifier' or 'mlm'"
        if step_type == 'classifier':
            step_kwargs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'concepts_values',
                                                                   'ignore_unknown']}
            loss_dict.update(self._predict_concepts_forward_step(return_predictions=False,
                                                                 return_probabilities=False,
                                                                 **step_kwargs))
        elif step_type == 'mlm':
            step_kwargs = {k: v for k, v in batch.items() if k in ['mlm_input_ids', 'mlm_attention_mask',
                                                                   'mlm_output_input_ids', 'orientations']}
            loss_dict.update(self._mlm_forward_step(**step_kwargs))
        return loss_dict

    def _init_loss_dict(self, eval_mode: bool = False,
                        batch_size: int = None,
                        return_keys: bool = False) -> Union[List[str], Dict[str, Any]]:
        concepts = self.model.language_masker.concepts
        predict_concepts = self.classifier.concepts
        if batch_size is None:
            none_list = None
        else:
            none_list = [None for _ in range(batch_size)]
        if eval_mode:
            prefixes = ['', 'masked_', 'generated_']
            loss_dict = {'example_id': none_list, 'text': none_list,
                         'masked_text': none_list, 'generated_text': none_list}
            for concept in concepts:
                loss_dict.update({f'concept_{concept}': none_list})
                if concept in self.model.concepts_to_control:
                    loss_dict.update({f'new_concept_{concept}': none_list,
                                      f'orientation_{concept}': none_list})
        else:
            prefixes = ['']
            loss_dict = {}
        loss_dict.update({'loss': torch.as_tensor(float('nan')),
                          'mlm_loss': torch.as_tensor(float('nan'))})
        for prefix in prefixes:
            loss_dict.update({f'{prefix}predict_loss': torch.as_tensor(float('nan')),
                              f'{prefix}predict_acc': torch.as_tensor(float('nan'))})
            loss_dict.update({f'{prefix}predict_{k}_loss': torch.as_tensor(float('nan'))
                              for k in predict_concepts})
            loss_dict.update({f'{prefix}predict_{k}_acc': torch.as_tensor(float('nan'))
                              for k in predict_concepts})
            if eval_mode:
                loss_dict.update({f'{prefix}prediction_{concept}': none_list for concept in predict_concepts})
        if return_keys:
            return list(loss_dict.keys())
        return loss_dict

    def training_step(self, batch, batch_idx: int) -> Dict[str, Any]:
        loss_dict = self._batch_forward_step(batch)
        self.log('train_loss', loss_dict['loss'], on_step=False, on_epoch=False, prog_bar=True, logger=True)
        return loss_dict

    def _epoch_end(self, outputs: List[Any], mode: str = 'train') -> Dict[str, Any]:
        epoch_loss_dict = {}
        if mode == 'train':
            losses_names = self._init_loss_dict(eval_mode=False, batch_size=None, return_keys=True)
            losses_names = [ln for ln in losses_names if ln.endswith(('loss', 'acc'))]
        else:
            losses_names = self._init_loss_dict(eval_mode=True, batch_size=None, return_keys=True)
            losses_names = [ln for ln in losses_names if ln.endswith(('loss', 'acc'))]
        for loss_name in losses_names:
            valid_losses = []
            for batch in outputs:
                if loss_name in batch and not torch.isnan(batch[loss_name]).any().item():
                    valid_losses.append(batch[loss_name])
            if len(valid_losses) > 0:
                loss_mean = torch.stack(valid_losses).mean()
            else:
                loss_mean = 0
            self.log(f'{mode}_mean_{loss_name}', loss_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            epoch_loss_dict[f'mean_{loss_name}'] = loss_mean
        return epoch_loss_dict

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self._epoch_end(outputs, mode='train')

    def _predict_concepts_to_eval_dict(self, input_ids: LongTensor,
                                       attention_mask: LongTensor,
                                       concepts_values: Dict[str, List[str]],
                                       ignore_unknown: bool = True,
                                       return_probabilities: bool = True,
                                       prefix: str = '') -> Dict[str, Any]:
        eval_dict = {}
        predict_output = self._predict_concepts_forward_step(input_ids=input_ids,
                                                             attention_mask=attention_mask,
                                                             concepts_values=concepts_values,
                                                             ignore_unknown=ignore_unknown,
                                                             return_predictions=True,
                                                             return_probabilities=return_probabilities,
                                                             with_encoder_grad=False)
        eval_dict.update({f'{prefix}{k}': v for k, v in predict_output.items() if k.endswith(('loss', 'acc'))})
        for concept, concept_predictions in predict_output['predictions'].items():
            eval_dict[f'{prefix}prediction_{concept}'] = concept_predictions
        if return_probabilities and 'probabilities' in predict_output:
            for concept, concept_probabilities in predict_output['probabilities'].items():
                eval_dict[f'{prefix}probabilities_{concept}'] = [{k: float(f'{v:.3f}') for k, v in p.items()}
                                                                 for p in concept_probabilities]
        return eval_dict

    def _eval_step(self, batch) -> Dict[str, Union[Tensor, List[str], int]]:
        if batch['step_type'] not in ['generation']:
            return self._batch_forward_step(batch)
        example_id = batch['example_id']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        concepts_values: Dict[str, List[str]] = batch['concepts_values']
        orientations: Dict[str, List[str]] = batch['orientations']
        new_concepts_values: Dict[str, List[str]] = batch['new_concepts_values']
        _concepts_values, orientations, _new_concepts_values = self.model._fix_inputs(
            len(example_id), concepts_values, orientations, new_concepts_values)
        # we use update to keep additional concepts which are not controlled
        concepts_values.update(_concepts_values)
        new_concepts_values.update(_new_concepts_values)
        for concept in concepts_values:
            if concept not in new_concepts_values:  # not a concept to control, we should not change it
                new_concepts_values[concept] = concepts_values[concept]
        mlm_input_ids = batch['mlm_input_ids']
        mlm_attention_mask = batch['mlm_attention_mask']
        return_probabilities = batch.get('return_probabilities', False)
        ignore_unknown = batch.get('ignore_unknown', True)
        generate_kwargs = {k: batch[k] for k in ['use_oriented_processor', 'num_beams', 'num_beam_groups',
                                                 'repetition_penalty', 'length_penalty', 'diversity_penalty',
                                                 'early_stopping','num_return_sequences',
                                                 'do_sample', 'temperature', 'top_k', 'top_p'] if k in batch}
        num_return_sequences = generate_kwargs.get('num_return_sequences', 1)
        generated_input_ids = self.model.generate_ids(input_ids=mlm_input_ids,
                                                      attention_mask=mlm_attention_mask,
                                                      orientations=orientations,
                                                      **generate_kwargs)
        input_text = self.model.decode_texts(input_ids)
        masked_text = self.model.decode_texts(mlm_input_ids,
                                              skip_special_tokens=False, remove_pad_and_eos=True)
        if self.model.language_masker.masked_output:
            generated_text = self.model.decode_texts(generated_input_ids,
                                                     skip_special_tokens=False, remove_pad_and_eos=True)
            generated_text = self.model.language_masker.combine_input_output(masked_text, generated_text)
        else:
            generated_text = self.model.decode_texts(generated_input_ids)
        generated_text = [self.model.processor.clean_spaces(t) for t in generated_text]
        eval_dict = self._init_loss_dict(eval_mode=True, batch_size=len(example_id), return_keys=False)
        eval_dict['example_id'] = example_id
        eval_dict['text'] = input_text
        eval_dict['masked_text'] = masked_text

        for concept, concept_values in concepts_values.items():
            eval_dict[f'concept_{concept}'] = concept_values
        for concept, concept_orientation in orientations.items():
            eval_dict[f'orientation_{concept}'] = concept_orientation
        for concept, concept_values in new_concepts_values.items():
            eval_dict[f'new_concept_{concept}'] = concept_values

        eval_dict.update(self._predict_concepts_to_eval_dict(input_ids=input_ids, attention_mask=attention_mask,
                                                             concepts_values=concepts_values,
                                                             ignore_unknown=ignore_unknown,
                                                             return_probabilities=return_probabilities,
                                                             prefix=''))
        eval_dict.update(self._predict_concepts_to_eval_dict(input_ids=mlm_input_ids, attention_mask=mlm_attention_mask,
                                                             concepts_values=concepts_values,
                                                             ignore_unknown=ignore_unknown,
                                                             return_probabilities=return_probabilities,
                                                             prefix='masked_'))
        # if we generated more than one sequence per example, we should repeat each list in eval_dict
        if num_return_sequences > 1:
            eval_dict = repeat_lists_in_dict(eval_dict, num_return_sequences)
            eval_dict['example_id'] = [f'{eid}{SEP}{i}' for i, eid in enumerate(eval_dict['example_id'])]
            new_concepts_values = repeat_lists_in_dict(new_concepts_values, num_return_sequences)
        eval_dict['generated_text'] = generated_text
        tokenized_generated = self.model.tokenize_texts(generated_text, return_tensors='pt')
        generated_input_ids = tokenized_generated['input_ids'].to(input_ids.device)
        generated_attention_mask = tokenized_generated['attention_mask'].to(attention_mask.device)
        # put UNK as concept_value if generated is less than 4 words or has less than 25% overlap with the input
        for i, (ie, ge) in enumerate(zip(eval_dict['text'], generated_text)):
            ie, ge = set(ie.split()), set(ge.split())
            if not (len(ie.intersection(ge)) >= len(ie) / 4 and len(ie) >= 4):
                for c in new_concepts_values:
                    new_concepts_values[c][i] = UNK
        eval_dict.update(self._predict_concepts_to_eval_dict(input_ids=generated_input_ids,
                                                             attention_mask=generated_attention_mask,
                                                             concepts_values=new_concepts_values,
                                                             ignore_unknown=ignore_unknown,
                                                             return_probabilities=return_probabilities,
                                                             prefix='generated_'))
        return eval_dict

    def _eval_epoch_end(self, outputs: List[Dict[str, Union[Tensor, List[str], int]]], mode: str):
        self._epoch_end(outputs, mode=mode)
        to_save_keys = self._init_loss_dict(eval_mode=True, batch_size=None, return_keys=True)
        to_save_keys = [k for k in to_save_keys if not k.endswith(('loss', 'acc'))]
        to_save_outputs = [subset_dict(batch, to_save_keys) for batch in outputs]
        to_save_outputs = concat_dicts(to_save_outputs)
        self.write_outputs(to_save_outputs, mode, to_csv=True)

    def validation_step(self, batch: BatchEncoding, batch_idx: int):
        return self._eval_step(batch)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self._eval_epoch_end(outputs, mode='validation')

    def test_step(self, batch: BatchEncoding, batch_idx: int):
        return self._eval_step(batch)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self._eval_epoch_end(outputs, mode='test')

    def write_outputs(self, outputs_dict: Dict[str, List[Any]], mode: str, to_csv: bool = False) -> None:
        output_dir = Path(self.hparams.output_dir)
        output_dir.mkdir(exist_ok=True)
        save_json(outputs_dict, output_dir / f"{mode}_outputs.json")
        if to_csv:
            try:
                df = pd.DataFrame(outputs_dict)
                df.to_csv(output_dir / f"{mode}_e{self.current_epoch}_outputs.csv", index=False)
            except Exception as e:
                print(f"Could not write outputs, encounter Exception at self.write_outputs(): "
                      f"{type(e)}:{e}")

    def configure_adamw_with_decay(self,
                                   weight_decay: float = None,
                                   lr: float = None,
                                   eps: float = None):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            # model (T5) parameters without bias and LayerNorm
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            # model (T5) bias and LayerNorm
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            # non model parameters without bias and LayerNorm
            {
                "params": [p for n, p in self.named_parameters()
                           if 'model.' not in n and not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay, "lr": lr * 5
            },
            # non model bias and LayerNorm
            {
                "params": [p for n, p in self.named_parameters()
                           if 'model.' not in n and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0, "lr": lr * 5
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)

    def configure_optimizers(self):
        optimizer = self.configure_adamw_with_decay(weight_decay=self.hparams.optimizer_weight_decay,
                                                    lr=self.hparams.optimizer_lr,
                                                    eps=self.hparams.optimizer_eps)
        if self.hparams.training_steps is not None and self.hparams.training_steps > 0:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.hparams.warmup_steps,
                                                        num_training_steps=self.hparams.training_steps)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            return {'optimizer': optimizer}

    @classmethod
    def load_from_checkpoint_workaround(cls, checkpoint_path, map_location=None, **kwargs):
        checkpoint_obj = torch.load(checkpoint_path)
        hparams_dict = checkpoint_obj['hyper_parameters']
        hparams = Namespace(**hparams_dict)
        model = cls.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                         map_location=map_location,
                                         hparams=hparams, **kwargs)
        return model
