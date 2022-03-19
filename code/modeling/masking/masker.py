from collections import defaultdict
from typing import Any
import re
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from scipy.stats import entropy as scipy_entropy
from collections import OrderedDict
from pathlib import Path
from typing import List, Union, Tuple, Dict

from modeling.masking.mapper import ConceptsMapping
from modeling.masking.text_processor import TextProcessor
from project_utils.constants import UNK, T5_MODEL_NAME
from project_utils.functions import sequence_indices, save_json, load_json


def prepare_mlm_input_output(splitted_text: List[str],
                             mask: List[bool]) -> Tuple[List[str], List[str]]:
    n = len(splitted_text)
    i = 0
    is_masked = False
    mask_cnt = 0
    non_mask_cnt = 0
    input_words, output_words = [], []
    while i < n:
        while i < n and mask[i] == True:
            is_masked = True
            output_words.append(splitted_text[i])
            i += 1
        if is_masked:
            input_words.append(f"<extra_id_{mask_cnt}>")
            mask_cnt += 1
        while i < n and mask[i] == False:
            is_masked = False
            input_words.append(splitted_text[i])
            i += 1
        if not is_masked:
            output_words.append(f"<extra_id_{non_mask_cnt}>")
            non_mask_cnt += 1
    return input_words, output_words


def mask_text(splitted_text: List[str],
              mask: List[bool] = None,
              probability: Union[float, List[float], np.ndarray] = None,
              rnp: Union[int, np.random.RandomState] = None,
              masked_output: bool = True) -> Tuple[List[str], List[str]]:
    assert probability is not None or mask is not None
    if mask is None:
        if rnp is None or isinstance(rnp, int):
            rnp = np.random.RandomState(rnp)
        sampled = rnp.uniform(0.0, 1.0, len(splitted_text))
        if not isinstance(probability, float):
            probability = np.array(probability)
        mask = sampled <= probability
    input_words, output_words = prepare_mlm_input_output(splitted_text, mask)
    if not masked_output:
        output_words = splitted_text
    return input_words, output_words


def mask_batch(splitted_texts: List[List[str]],
               masks: List[List[bool]] = None,
               probabilities: Union[float, List[Union[float, List[float], np.ndarray]]] = None,
               rnp: Union[int, np.random.RandomState] = None,
               masked_output: bool = True) -> Tuple[List[List[str]], List[List[str]]]:
    assert probabilities is not None or masks is not None
    if masks is None:
        masks = [None for _ in range(len(splitted_texts))]
    if probabilities is None or isinstance(probabilities, float):
        probabilities = [probabilities for _ in range(len(splitted_texts))]
    if rnp is None or isinstance(rnp, int):
        rnp = np.random.RandomState(rnp)
    inputs, outputs = [], []
    for splitted_text, mask, probability in zip(splitted_texts, masks, probabilities):
        input_words, output_words = mask_text(splitted_text, mask, probability, rnp, masked_output)
        inputs.append(input_words)
        outputs.append(output_words)
    return inputs, outputs


def combine_input_output(masked_input: List[str], masked_outputs: List[str]) -> List[str]:
    assert len(masked_input) == len(masked_outputs)
    combined = []
    for input_text, output_text in zip(masked_input, masked_outputs):
        if output_text.startswith('<extra_id_'):
            first_text = input_text
            second_text = output_text
        else:
            first_text = output_text
            second_text = input_text
        first_text = list(filter(None, re.split(r' *<extra_id_\d+> *', first_text)))
        second_text = list(filter(None, re.split(r' *<extra_id_\d+> *', second_text)))
        num = min(len(first_text), len(second_text))
        combined_text = [''] * (num * 2)
        combined_text[::2] = first_text[:num]
        combined_text[1::2] = second_text[:num]
        combined_text.extend(first_text[num:])
        combined_text.extend(second_text[num:])
        combined.append(' '.join(combined_text))
    return combined


class ConceptsModeling:
    def __init__(self,
                 text_processor: TextProcessor = None,
                 n_grams: int = 3,
                 **kwargs):
        if isinstance(text_processor, TextProcessor):
            self.processor = text_processor
        else:
            processor_kwargs = {k: v for k, v in kwargs.items() if k in TextProcessor.init_kwargs}
            processor_kwargs['tokenizer'] = processor_kwargs.get('tokenizer', T5_MODEL_NAME)
            self.processor = TextProcessor(**processor_kwargs)
        self.n_grams = n_grams
        self.concepts_mapping = kwargs.get('concepts_mapping', ConceptsMapping())
        self.frequencies = kwargs.get('frequencies', {})
        self.concepts_frequencies = kwargs.get('concepts_frequencies', {})
        self.orientations = kwargs.get('orientations', {})
        self.orientations_cnt = kwargs.get('orientations_cnt', {})
        self.probabilities = kwargs.get('probabilities', {})
        for concept in self.concepts_mapping:
            for value in self.concepts_mapping.concept_values(concept):
                self.new_concept_value(concept, value)

    @property
    def concepts(self):
        return self.concepts_mapping.concepts

    def _get_kwargs(self) -> Dict:
        kwargs = {}
        kwargs['tokenizer'] = self.processor.tokenizer_name
        kwargs['max_seq_len'] = self.processor.max_seq_len
        kwargs['skip_special_tokens'] = self.processor.skip_special_tokens
        kwargs['clean_up_tokenization_spaces'] = self.processor.clean_up_tokenization_spaces
        kwargs['n_grams'] = self.n_grams
        kwargs['concepts_mapping'] = {c: v for c, v in self.concepts_mapping.items()}
        kwargs['frequencies'] = self.frequencies
        kwargs['probabilities'] = self.probabilities
        kwargs['concepts_frequencies'] = self.concepts_frequencies
        kwargs['orientations'] = self.orientations
        kwargs['orientations_cnt'] = self.orientations_cnt
        return kwargs

    @classmethod
    def from_json(cls, json_path: Union[str, Path],
                  **additional_kwargs):
        kwargs = load_json(json_path)
        cm = ConceptsMapping()
        cm.update(kwargs.pop('concepts_mapping'))
        kwargs['concepts_mapping'] = cm
        kwargs.update(additional_kwargs)
        return cls(**kwargs)

    def to_json(self, json_path: Union[str, Path], **kwargs_to_add):
        json_dict = self._get_kwargs()
        json_dict.update(kwargs_to_add)
        save_json(json_dict, json_path)
        save_json(json_dict, json_path)

    @property
    def concepts_orientations(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        return self.orientations

    @property
    def concepts_ordered_orientations(self) -> Dict[str, List[str]]:
        return {c: self.concept_ordered_orientations(c) for c in self.concepts}

    def orientation_to_value(self, concept: str, orientation: str) -> str:
        orientation = orientation if orientation is not None else UNK
        if orientation == UNK:
            return UNK
        for value, orientations in self.orientations[concept].items():
            if orientation in orientations:
                return value
        return UNK

    def orientation_to_index(self, concept: str, value: Union[str, None], orientation: str) -> int:
        orientation = orientation if orientation is not None else UNK
        value = value if value is not None else self.orientation_to_value(concept, orientation)
        if f'{value} {orientation}' in self.orientations[concept][value]:
            return self.orientations[concept][value][f'{value} {orientation}']
        return self.orientations[concept][value][orientation]

    def concept_orientations(self, concept: str) -> Dict[str, Dict[str, Dict[str, int]]]:
        return self.orientations[concept]

    def concept_ordered_orientations(self, concept: str) -> List[str]:
        orientations = self.orientations[concept]
        return [o for o, _ in sorted([(o, i) for v, ois in orientations.items()
                                      for o, i in ois.items()], key=lambda oi: oi[1])]

    def concept_value_orientations(self, concept: str, value: str) -> Dict[str, int]:
        value = value if value is not None else UNK
        return self.orientations[concept][value]

    def concept_value_ordered_orientations(self, concept: str, value: str) -> List[str]:
        value = value if value is not None else UNK
        orientations = self.orientations[concept][value]
        return [o for o, _ in sorted([(o, i) for o, i in orientations.items()], key=lambda oi: oi[1])]

    def new_concept_value(self, concept: str,
                          value: str):
        assert concept is not None
        self.concepts_mapping.add_concept_value(concept, value)
        # new concept
        if concept not in self.orientations:
            self.concepts_frequencies[concept] = [0]
            self.orientations_cnt[concept] = 0
            self.orientations[concept] = {UNK : {UNK: 0}}
            for word in self.frequencies:
                self.frequencies[word][concept] = [0]
        # new value
        if value is not None and value != UNK and value not in self.orientations[concept]:
            self.concepts_frequencies[concept] += [0]
            self.orientations_cnt[concept] += 1
            self.orientations[concept][value] = {value: self.orientations_cnt[concept]}
            for word in self.frequencies:
                self.frequencies[word][concept] += [0]

    def new_word(self, word: str):
        if word not in self.frequencies:
            self.frequencies[word] = {}
            for concept in self.concepts_mapping:
                n = len(self.concepts_mapping.concept_values(concept))
                self.frequencies[word][concept] = [0] * n

    def add_concept_value(self, concept: str,
                          value: str):
        self.new_concept_value(concept, value)
        self.concepts_frequencies[concept][self.concepts_mapping.value_to_index(concept, value)] += 1

    def stem_word(self, word: str):
        return self.processor.stem(word)

    def add_word(self, word: str,
                 concept: str,
                 value: str):
        word = self.stem_word(word)
        self.new_concept_value(concept, value)
        self.new_word(word)
        value_index = self.concepts_mapping.value_to_index(concept, value)
        self.frequencies[word][concept][value_index] += 1

    def add_texts(self, texts: List[str],
                  concepts_values: Dict[str, List[str]]):
        for i, text in enumerate(texts):
            words = self.processor.get_words(text, n_grams=list(range(1, self.n_grams + 1)), stem=True)
            words = list(OrderedDict.fromkeys(words))  # remove duplicates
            for concept, values in concepts_values.items():
                value = values[i]
                self.add_concept_value(concept, value)
                for word in words:
                    self.add_word(word, concept, value)

    def filter_minimum_occurrences(self, min_occurrences: int):
        to_filter = []
        for word in self.frequencies:
            word_max_freq = max([sum(self.frequencies[word][concept]) for concept in self.frequencies[word]])
            if word_max_freq < min_occurrences:
                to_filter.append(word)
        for word in to_filter:
            self.frequencies.pop(word)

    def add_dataset(self, dataset: Dataset,
                    text_column: str,
                    concept_columns: List[str],
                    batch_size: int = 5000,
                    min_occurrences: int = None):
        for i in tqdm(range(int(np.ceil(len(dataset) / batch_size)))):
            texts = dataset[text_column][i * batch_size: (i + 1) * batch_size]
            concepts_values = {concept: dataset[concept][i * batch_size: (i + 1) * batch_size]
                               for concept in concept_columns}
            self.add_texts(texts, concepts_values)
        if min_occurrences is not None:
            self.filter_minimum_occurrences(min_occurrences)

    def add_orientation(self, concept: str, value: str, orientation: str) -> str:
        self.new_concept_value(concept, value)
        if orientation in self.orientations[concept][value]:  # already in
            return UNK
        # looking for all values that might have orientation (or its extended version 'value orientation').
        values = []
        for v, os in self.orientations[concept].items():
            if orientation in os or f'{v} {orientation}' in os:
                values.append(v)
        if len(values) > 1:  # there are already values with this extended orientation, so we need to extend.
            orientation = f'{value} {orientation}'
        elif len(values) == 1:  # there is a value with this orientation, we need to extend both.
            old_value = values[0]
            old_orientation_index = self.orientations[concept][old_value].pop(orientation)
            self.orientations[concept][old_value][f'{old_value} {orientation}'] = old_orientation_index
            orientation = f'{value} {orientation}'
        else:  # no need to extend, this is a new orientation.
            pass
        self.orientations[concept][value][orientation] = self.orientations_cnt[concept]
        self.orientations_cnt[concept] += 1
        return orientation

    def reset_orientations(self, concept: str):
        self.orientations[concept] = {v: {} for v in self.concepts_mapping.concept_values(concept)}
        self.orientations[concept][UNK] = {UNK: 0}  # adding the unknown orientation
        self.orientations_cnt[concept] = 1

    def add_orientations(self, concept: str, value: str, orientations: List[str]):
        self.add_concept_value(concept, value)
        for orientation in orientations:
            self.add_orientation(concept, value, orientation)

    def update_cached_probabilities(self, smoothing: Union[str, int, float, List[int]] = None,
                                    batch_size: int = None,
                                    device: str = None):
        for word in self.frequencies:
            if word not in self.probabilities:
                self.probabilities[word] = {}
            for concept in self.concepts:
                self.probabilities[word][concept] = self.word_probabilities(word, concept,
                                                                            word_given_concept=True,
                                                                            smoothing=smoothing,
                                                                            return_np=True,
                                                                            using_cached=False).tolist()

    @staticmethod
    def neg_ent_score(concept_probabilities: Dict[str, float],
                      unknown_key: str = UNK) -> float:
        unknown_prob = concept_probabilities.get(unknown_key, 0.0)
        probs = np.array([p for k, p in concept_probabilities.items() if k != unknown_key])
        probs = (probs / np.sum(probs)) if np.sum(probs) != 0 else probs  # normalize the known probabilities
        entropy = scipy_entropy(probs) / max(np.log(len(probs)), 1e-8)
        return 1.0 - (unknown_prob + (1.0 - unknown_prob) * entropy)

    def top_values_words(self, top: int = 15,
                         with_value_name: bool = True,
                         top_occurrences_threshold: int = None,
                         concepts: List[str] = None) -> Dict[str, Dict[str, List[str]]]:
        concepts = concepts if concepts is not None else self.concepts
        top_words = {concept: {v: [] for v in self.concepts_mapping.concept_values(concept) if v != UNK}
                     for concept in concepts}
        tokenizer_words = set(self.processor.stemmed_vocab.values())
        for word in self.frequencies:
            if word not in tokenizer_words:
                continue
            values_scores = []
            neg_ent_scores = {}
            word_p = {c: self.word_probabilities(word, c, smoothing=1) for c in concepts}
            freqs = {c: sum([f for v, f in self.word_frequencies(word, c, return_np=False).items() if v != UNK])
                     for c in concepts}
            for c, probs in word_p.items():
                neg_ent_scores[c] = self.neg_ent_score(probs)
                values_scores.extend([(p * neg_ent_scores[c], c, v) for v, p in probs.items()])
            if top_occurrences_threshold is not None:
                values_scores = [x for x in values_scores if freqs[x[1]] >= top_occurrences_threshold]
                if len(values_scores) == 0:
                    continue
            values_scores = sorted(values_scores, key=lambda x: x[0], reverse=True)
            first_s, first_concept, first_value = values_scores[0]
            mean_s = sum([x[0] for x in values_scores[1:]]) / (len(values_scores) - 1)
            score = np.log1p(freqs[first_concept]) * (first_s - mean_s)
            top_words[first_concept][first_value].append((word, score))
        for concept in top_words:
            for v, scores in top_words[concept].items():
                top_words[concept][v] = [w for w, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:top]]
                if with_value_name:
                    stemmed_value = self.stem_word(v)
                    if stemmed_value not in top_words[concept][v]:
                        top_words[concept][v] = [stemmed_value] + top_words[concept][v][:-1]
                    else:
                        top_words[concept][v] = [stemmed_value] + [w for w in top_words[concept][v] if
                                                                   w != stemmed_value]
        return top_words

    def add_orientations_top_words(self, n_orientations: int = 4,
                                   with_value_name: bool = True,
                                   top_occurrences_threshold: int = None,
                                   concepts: List[str] = None):
        top_words = self.top_values_words(n_orientations, with_value_name, top_occurrences_threshold, concepts)
        for concept in top_words:
            self.reset_orientations(concept)
            for value, words in top_words[concept].items():
                self.add_orientations(concept, value, orientations=words)

    def word_frequencies(self, word: str,
                         concept: str,
                         return_np: bool = False) -> Union[Dict[str, float], np.array]:
        assert concept in self.concepts
        word = self.stem_word(word)
        if word not in self.frequencies:
            freq = [0] * len(self.concepts_mapping.concept_values(concept))
        else:
            freq = self.frequencies[word][concept]
        if return_np:
            return np.array(freq)
        else:
            return {self.concepts_mapping.index_to_value(concept, i): f for i, f in enumerate(freq)}

    def word_cached_probabilities(self, word: str,
                                  concept: str,
                                  return_np: bool = False) -> Union[Dict[str, float], np.array]:
        assert concept in self.concepts
        word = self.stem_word(word)
        if word not in self.probabilities:  # uniform probability
            n_values = len(self.concepts_mapping.concept_values(concept))
            prob = [1/n_values] * n_values
        else:
            prob = self.probabilities[word][concept]
        if return_np:
            return np.array(prob)
        else:
            return {self.concepts_mapping.index_to_value(concept, i): f for i, f in enumerate(prob)}

    def word_probabilities(self, word: str,
                           concept: str,
                           word_given_concept: bool = False,
                           values_to_ignore: Union[str, None, List[str]] = UNK,
                           smoothing: Union[str, int, List[int]] = None,
                           return_np: bool = False,
                           using_cached: bool = True) -> Union[np.array, Dict[str, float]]:
        assert concept in self.concepts
        if isinstance(values_to_ignore, str):
            values_to_ignore = [values_to_ignore]
        if using_cached and len(self.probabilities) != 0:
            probs = self.word_cached_probabilities(word, concept, return_np=True)
        else:
            smoothing = smoothing if smoothing is not None else 1
            if isinstance(smoothing, str) and smoothing == 'n':
                smoothing = len(word.split())
            elif isinstance(smoothing, (list, tuple)):
                smoothing = smoothing[len(word.split())-1]
            freq = self.word_frequencies(word, concept, return_np=True)  # if word / concept not exist - freq is zeroes
            if freq.sum() == 0:  # word / concept is not exist, uniform probability
                probs = freq + 1 / freq.shape[0]
            else:
                probs = freq + smoothing / freq.shape[0]
            if word_given_concept:
                # probability of token given concept-value, then normalizing (sum to 1)
                cfreqs = np.array(self.concepts_frequencies[concept])
                probs = probs / np.where(cfreqs == 0, 1, cfreqs)

        if values_to_ignore is not None and len(values_to_ignore) > 0:
            values_indices = [self.concepts_mapping.value_to_index(concept, v) for v in values_to_ignore]
            probs[values_indices] = 0.0
            # normalizing (sum to 1)
        probs_sum = probs.sum()
        if probs_sum != 0:  # else - it is already zeros
            probs /= probs_sum
        if return_np:
            return probs
        return {v: float(prob) for prob, v in zip(probs, self.concepts_mapping.concept_values(concept))}

    def text_probabilities(self, text: str,
                           concept: Union[str, List[str]],
                           word_given_concept: bool = False,
                           values_to_ignore: Union[str, None, List[str]] = UNK,
                           smoothing: Union[str, int, float, List[Union[int, float]]] = None,
                           n_grams: Union[int, List[int]] = None,
                           using_cached: bool = True) -> List[Tuple[str, Dict]]:
        smoothing = smoothing if smoothing is not None else 1
        n_grams = n_grams if n_grams is not None else list(range(1, self.n_grams + 1))
        words = list(OrderedDict.fromkeys(self.processor.get_words(text, n_grams, stem=False)))  # remove duplicates
        probabilities = []
        for word in words:
            if isinstance(concept, str):
                probabilities.append((word, self.word_probabilities(word, concept,
                                                                    word_given_concept=word_given_concept,
                                                                    values_to_ignore=values_to_ignore,
                                                                    smoothing=smoothing, return_np=False,
                                                                    using_cached=using_cached)))
            else:
                word_p = {c: self.word_probabilities(word, c, word_given_concept=word_given_concept,
                                                     values_to_ignore=values_to_ignore, smoothing=smoothing,
                                                     return_np=False, using_cached=using_cached) for c in concept}
                probabilities.append((word, word_p))
        return probabilities  # [(word, {v: p, }), ] or [(word, {c: {v: p, }, }), ]


class LanguageMasker(ConceptsModeling):
    def __init__(self,
                 text_processor: TextProcessor = None,
                 n_grams: int = 3,
                 masked_output: bool = False,
                 threshold: float = 0.08,
                 top_n: str = 0.15,
                 noise: float = 0.05,
                 seed: int = None,
                 **kwargs):
        super(LanguageMasker, self).__init__(text_processor, n_grams, **kwargs)
        self.masked_output = masked_output
        self.threshold = threshold
        self.top_n = top_n
        self.noise = noise
        self.seed = seed
        self.rnp = np.random.RandomState(self.seed)
        self.masked_words = kwargs.get('masked_words', {})

    def _get_kwargs(self) -> Dict:
        kwargs = super()._get_kwargs()
        kwargs['masked_output'] = self.masked_output
        kwargs['threshold'] = self.threshold
        kwargs['top_n'] = self.top_n
        kwargs['noise'] = self.noise
        kwargs['seed'] = self.seed
        kwargs['masked_words'] = self.masked_words
        return kwargs

    @staticmethod
    def masking_score(concept_probabilities: Dict[str, float],
                      value_to_mask: str,
                      new_value: str,
                      n_concepts: int = 1,
                      unknown_key: str = UNK) -> float:
        cp = concept_probabilities
        value_to_mask = value_to_mask if value_to_mask in cp else UNK
        new_value = new_value if new_value in cp else UNK
        neg_ent_score = cp.get('neg_ent_score', LanguageMasker.neg_ent_score(cp, unknown_key))
        known_ps = [pv for cv, pv in cp.items() if cv != UNK]
        if len(known_ps) == 0:
            max_p = 1.0 / n_concepts
            min_p = 1.0 / n_concepts
        else:
            max_p = max(known_ps)
            min_p = min(known_ps)
        value_p = cp[value_to_mask] if value_to_mask != UNK else max_p
        new_value_p = cp[new_value] if value_to_mask != UNK else min_p
        score = neg_ent_score * (value_p - new_value_p)
        # in the case the new value is like the old value (mostly when controlling multiple concepts),
        # we would like to provide a negative score to terms which correlate with the old/nbew value
        # so they won't be masked.
        if value_to_mask == new_value:
            score = score - neg_ent_score * cp[new_value] / n_concepts
        return score

    def update_cached_masked_words(self, using_cached: bool = True,
                                   smoothing: Union[str, int, float] = None,
                                   threshold: float = None):
        threshold = threshold if threshold is not None else self.threshold
        dependencies = {word: [] for word in self.frequencies}
        splitted_vocab = {word: word.split() for word in dependencies}
        n_concepts = len(self.concepts)
        for word, splitted_words in splitted_vocab.items():
            for splitted_word in splitted_words:
                if word == splitted_word:
                    continue
                dependencies[splitted_word].append(word)
        masked_words = {}
        for concept in self.concepts:
            masked_words[concept] = {}
            for word in dependencies:
                cp = self.word_probabilities(word, concept, word_given_concept=True,
                                             smoothing=smoothing, using_cached=using_cached)
                for value in self.concepts_mapping.concept_values(concept):
                    if value not in masked_words[concept]:
                        masked_words[concept][value] = set([])
                    # not adding word if already partial gram of it is masked
                    skip_word = False
                    for splitted_word in splitted_vocab[word]:
                        if splitted_word in masked_words[concept][value]:
                            skip_word = True
                    if skip_word:
                        continue
                    score = self.masking_score(cp, value_to_mask=value, new_value=UNK, n_concepts=n_concepts)
                    if score > threshold:
                        masked_words[concept][value].add(word)
                        for dep_word in dependencies[word]:
                            if dep_word in masked_words[concept][value]:
                                masked_words[concept][value].remove(dep_word)
        for concept in masked_words:
            for value in masked_words[concept]:
                masked_words[concept][value] = list(masked_words[concept][value])
        self.masked_words = masked_words

    def mask_splitted_texts(self, splitted_texts: List[List[str]],
                            masks: List[List[bool]] = None,
                            probabilities: Union[float, List[Union[float, List[float], np.ndarray]]] = None,
                            masked_output: bool = False,
                            return_tensors: str = 'pt',
                            device: str = None) -> Dict[str, Any]:
        mlm_inputs, mlm_outputs = mask_batch(splitted_texts, masks=masks, probabilities=probabilities,
                                             masked_output=masked_output, rnp=self.rnp)
        mlm_inputs = [self.processor.preprocess_text(' '.join(t), clean_spaces=True) for t in mlm_inputs]
        mlm_outputs = [self.processor.preprocess_text(' '.join(t), clean_spaces=True) for t in mlm_outputs]
        tokenized_inputs = self.processor.tokenize_texts(mlm_inputs, return_tensors=return_tensors)
        tokenized_outputs = self.processor.tokenize_texts(mlm_outputs, return_tensors=return_tensors)
        mlm_input_ids = tokenized_inputs['input_ids']
        mlm_attention_mask = tokenized_inputs['attention_mask']
        mlm_output_input_ids = tokenized_outputs['input_ids']
        if return_tensors == 'pt' and device is not None:
            mlm_input_ids = mlm_input_ids.to(device)
            mlm_attention_mask = mlm_attention_mask.to(device)
            mlm_output_input_ids = mlm_output_input_ids.to(device)
        return {'mlm_input_ids': mlm_input_ids,
                'mlm_attention_mask': mlm_attention_mask,
                'mlm_output_input_ids': mlm_output_input_ids,
                'masked_text': mlm_inputs,
                'masked_output_text': mlm_outputs}

    def regular_mlm_mask(self, texts: List[str],
                         mlm_probability: float = 0.15,
                         masked_output: bool = None,
                         return_tensors: str = 'np',
                         device: str = None) -> Dict[str, Any]:
        masked_output = masked_output if masked_output is not None else self.masked_output
        splitted_texts = [t.split() for t in texts]
        masked_batch = self.mask_splitted_texts(splitted_texts=splitted_texts, probabilities=mlm_probability,
                                                return_tensors=return_tensors,
                                                device=device, masked_output=masked_output)
        masked_batch['probabilities'] = None
        return masked_batch

    def words_concepts_modeling_mask(self, texts: List[str],
                                     concepts_values: Dict[str, List[str]],
                                     new_concepts_values: Dict[str, List[str]],
                                     masked_output: bool = None,
                                     threshold: float = None,
                                     top_n: Union[int, float] = None,
                                     noise: Union[float, int, bool] = None,
                                     smoothing: Union[str, int, float] = None,
                                     n_grams: Union[int, List[int]] = None,
                                     using_cached: bool = True,
                                     return_tensors: str = 'np',
                                     device: str = None,
                                     return_probabilities: bool = False):
        masked_output = masked_output if masked_output is not None else self.masked_output
        threshold = threshold if threshold is not None else self.threshold
        top_n = top_n if top_n is not None else self.top_n
        noise = noise if noise is not None else self.noise
        smoothing = smoothing if smoothing is not None else 1
        n_grams = n_grams if n_grams is not None else list(range(1, self.n_grams + 1))
        concepts = list(new_concepts_values.keys())
        n_concepts = len(concepts)
        splitted_texts, masks = [], []
        probabilities = {concept_name: [] for concept_name in concepts}
        for i, text in enumerate(texts):
            text_mask_scores = {}
            text_values, new_values, text_probabilities = {}, {}, {}
            for concept in concepts:
                text_values[concept] = concepts_values[concept][i]
                new_values[concept] = new_concepts_values[concept][i]
                text_probabilities[concept] = []
            text_words = self.processor.get_words(text, n_grams=1, stem=False)
            text_p = self.text_probabilities(text, concept=concepts, word_given_concept=True,
                                             values_to_ignore=UNK, smoothing=smoothing,
                                             n_grams=n_grams, using_cached=using_cached)
            for word, probs in text_p:
                if word in text_mask_scores:
                    continue
                word_mask_scores = {}
                for concept in concepts:
                    value = text_values[concept]
                    new_value = new_values[concept]
                    wcp = probs[concept]
                    wcp['mask'] = self.masking_score(wcp, value, new_value, n_concepts, unknown_key=UNK)
                    text_probabilities[concept].append((word,
                                                        {cv: float(f'{prob:.3f}') for cv, prob in wcp.items()}))
                    word_mask_scores[concept] = wcp['mask']
                text_mask_scores[word] = np.sum(np.array(list(word_mask_scores.values())))
            text_mask_scores = sorted(list(text_mask_scores.items()), reverse=True, key=lambda x: x[1])
            masked_words = {}
            masked_indices = defaultdict(list)
            n_words = len(text_words)
            mask = np.array([False] * n_words)
            if top_n is not None:
                text_top_n = int(top_n) if top_n >= 1.0 else int(top_n * n_words)
            else:
                text_top_n = 0
            for word, score in text_mask_scores:
                # stopping criteria for concepts_modeling.
                if score < threshold and len(masked_indices) >= text_top_n:
                    break
                # we don't use n-grams if we already masked one of their words (grams)
                continue_word = False
                word_n_grams = word.split()
                for masked_word in masked_words:
                    if len(sequence_indices(word_n_grams, masked_word.split())) > 0:
                        continue_word = True
                if continue_word:
                    continue

                list_of_indices_to_mask = sequence_indices(text_words, word.split())
                indices_to_mask = set([x for sub in list_of_indices_to_mask for x in sub])
                # concepts_modeling criteria
                if score >= threshold or len(set(masked_indices).union(indices_to_mask)) <= text_top_n:
                    # remove n-grams containing this word from the masked words
                    for masked_word in set(masked_words):
                        if len(sequence_indices(masked_word.split(), word_n_grams)) > 0:
                            indices_to_remove = masked_words.pop(masked_word)
                            for index in indices_to_remove:
                                if index in masked_indices:
                                    index_words = masked_indices[index]
                                    if len([True for ie in index_words if word in ie]) == len(index_words):
                                        masked_indices.pop(index)
                    # add the word and its indices
                    masked_words[word] = indices_to_mask
                    for index in indices_to_mask:
                        masked_indices[index].append(word)
            masked_indices = list(masked_indices.keys())
            if noise is not None and noise > 0.0:
                text_noise = int(noise) if noise >= 1.0 else int(noise * n_words)
                not_masked_indices = [ind for ind in range(n_words) if ind not in masked_indices]
                if text_noise > 0 and len(not_masked_indices) > 0:
                    text_noise = min(text_noise, len(not_masked_indices))
                    masked_indices += self.rnp.choice(not_masked_indices, text_noise, replace=False).tolist()
            mask[masked_indices] = True
            splitted_texts.append(text_words)
            mask = list(mask)
            masks.append(mask)
            if return_probabilities:
                for concept, text_concept_probabilities in text_probabilities.items():
                    probabilities[concept].append(text_concept_probabilities)
        masked_batch = self.mask_splitted_texts(splitted_texts=splitted_texts, masks=masks,
                                                return_tensors=return_tensors, device=device,
                                                masked_output=masked_output)
        masked_batch['probabilities'] = probabilities if return_probabilities else None
        return masked_batch

    @staticmethod
    def combine_input_output(masked_input: List[str], masked_outputs: List[str]) -> List[str]:
        return combine_input_output(masked_input, masked_outputs)
