from collections import defaultdict
from string import punctuation
from typing import List, Union, Tuple

import nltk
from nltk import StemmerI, SnowballStemmer
from transformers import PreTrainedTokenizer, AutoTokenizer
from project_utils.constants import MAX_SEQ_LEN, T5_MODEL_NAME


class TextProcessor:
    new_clause_puncs = {'.', '!', '?', ',', ';', ':', '"', '(', ')', '[', ']', '{', '}', '\n'}
    punctuations = set(punctuation)
    init_kwargs = {'tokenizer', 'stemmer', 'max_seq_len', 'skip_special_tokens', 'clean_up_tokenization_spaces'}

    def __init__(self,
                 tokenizer: Union[str, PreTrainedTokenizer] = T5_MODEL_NAME,
                 stemmer: StemmerI = None,
                 max_seq_len: int = MAX_SEQ_LEN,
                 skip_special_tokens: bool = True,
                 clean_up_tokenization_spaces: bool = True):
        """

        :param tokenizer:
        :param stemmer:
        :param max_seq_len:
        :param skip_special_tokens:
        :param clean_up_tokenization_spaces:
        """
        self.tokenizer_name = None
        if tokenizer is None:
            tokenizer = T5_MODEL_NAME
        if isinstance(tokenizer, str):
            self.tokenizer_name = tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.tokenizer = tokenizer
        if not isinstance(stemmer, StemmerI):
            try:
                stemmer = SnowballStemmer("english", ignore_stopwords=True)
            except LookupError as e:
                nltk.download('stopwords')
                stemmer = SnowballStemmer("english", ignore_stopwords=True)
        self.stemmer = stemmer
        self.max_seq_len = max_seq_len
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self._init_stemmed_vocab()

    def _init_stemmed_vocab(self):
        self.stemmed_vocab = {}
        stemmed_vocab = defaultdict(list)
        space_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer('the')['input_ids'][0]).replace('the', '')
        for k, v in self.tokenizer.vocab.items():
            k = k.replace(space_token, '').lower()
            stemmed_vocab[self.stem(k)].append(k)
        self.stemmed_vocab = {k: sorted(v, key=lambda w: len(w))[0] for k, v in stemmed_vocab.items()}

    def stem(self, word: str):
        grams = []
        for w in word.strip().split():
            w = self.stemmer.stem(w)
            if w.endswith('ies'):
                w = w[:-3] + 'y'
            elif w.endswith('ves'):
                w = w[:-3] + 'f'
            w = self.stemmed_vocab.get(w, w)
            grams.append(w)
        return ' '.join(grams)

    def tokenize_text(self, text: str):
        return self.tokenizer(text)['input_ids'][:-1]

    def tokenize_texts(self, batch_texts: List[str],
                       return_tensors: str = 'pt',
                       padding: Union[bool, str] = 'max_length',
                       truncation: bool = True,
                       max_length: int = None,
                       return_attention_mask: bool = True):
        max_length = max_length if max_length is not None else self.max_seq_len
        max_length = None if max_length <= 0 else max_length
        return self.tokenizer(batch_texts,
                              is_split_into_words=False,
                              padding=padding,
                              truncation=truncation,
                              max_length=max_length,
                              return_tensors=return_tensors,
                              return_attention_mask=return_attention_mask).data

    def decode_texts(self, batch_sequences,
                     skip_special_tokens: bool = None,
                     clean_up_tokenization_spaces: bool = None):
        skip_special_tokens = skip_special_tokens if skip_special_tokens is not None \
            else self.skip_special_tokens
        clean_up_tokenization_spaces = clean_up_tokenization_spaces if clean_up_tokenization_spaces is not None \
            else self.clean_up_tokenization_spaces
        return self.tokenizer.batch_decode(batch_sequences,
                                           skip_special_tokens=skip_special_tokens,
                                           clean_up_tokenization_spaces=clean_up_tokenization_spaces)

    @staticmethod
    def clean_spaces(text: str) -> str:
        text = (
            text.replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re")
                .replace('  ', ' ')
        )
        return text

    def remove_punctuation_prefix_suffix(self, word: str,
                                         return_tuple: bool = False) -> Union[str, Tuple[str, str, str]]:
        i, n = 0, len(word)
        while i < n and word[i] in self.punctuations:
            i += 1
        prefix, word = word[:i], word[i:]
        j, n = 1, len(word)
        while j <= n and word[-j] in self.punctuations:
            j += 1
        word, suffix = word[:-j + 1], word[-j + 1:]
        if j == 1:
            word, suffix = suffix, word
        if return_tuple:
            return prefix, word, suffix
        else:
            return word

    def split(self, text: str) -> List[str]:
        words = []
        for w in text.split():
            prefix, w, suffix = self.remove_punctuation_prefix_suffix(w, return_tuple=True)
            if prefix != '':
                words.append(prefix)
            if w != '':
                words.append(w)
            if suffix != '':
                words.append(suffix)
        return words

    def preprocess_text(self, text: Union[str, List[str]],
                        lower: bool = False,
                        space_between_punctuations: bool = False,
                        stem: bool = False,
                        clean_spaces: bool = False,
                        tokenize_decode: bool = False,
                        max_length: int = None):
        return_str = isinstance(text, str)
        if return_str:
            text = [text]
        if lower:
            text = [t.lower() for t in text]
        if space_between_punctuations:
            text = [' '.join(self.split(t.replace('-', ' - '))) for t in text]
        if stem:
            text = [' '.join([w if not stem else self.stem(w) for w in t.split()]) for t in text]
        if clean_spaces:
            text = [self.clean_spaces(t) for t in text]
        if tokenize_decode:
            tokenized = self.tokenize_texts(text, padding=True,
                                            truncation=True, max_length=max_length, return_tensors="np",
                                            return_attention_mask=False)['input_ids']
            text = self.decode_texts(tokenized, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
        if return_str:
            return text[0]
        else:
            return text

    @staticmethod
    def create_n_grams(list_of_words: List[str],
                       n_grams: int) -> List[str]:
        list_of_words = list(list_of_words)
        size = len(list_of_words)
        words = []
        for i in range(n_grams, size + 1):
            words.append(' '.join(list_of_words[i - n_grams:i]))
        return words

    def get_words(self, text: str,
                  n_grams: Union[int, List[int]] = 1,
                  stem: bool = False) -> List[str]:
        if isinstance(n_grams, int):
            n_grams = [n_grams]
        elif isinstance(n_grams, (list, tuple)):
            n_grams = list(n_grams)
        else:
            raise ValueError(f"`n_grams` must be an int or a list of ints.")

        words = []
        clause = []
        for w in text.split():
            prefix, w, suffix = self.remove_punctuation_prefix_suffix(w, return_tuple=True)
            if len(prefix) > 1 or prefix in self.new_clause_puncs:
                for i in n_grams:
                    words += self.create_n_grams(clause, n_grams=i)
                words += [prefix]
                clause = []
            elif prefix != '':
                clause.append(prefix)
            if w != '':
                if stem:
                    w = self.stem(w)
                clause.append(w)
            if len(suffix) > 1 or suffix in self.new_clause_puncs:
                for i in n_grams:
                    words += self.create_n_grams(clause, n_grams=i)
                words.append(suffix)
                clause = []
            elif suffix != '':
                clause.append(suffix)
        for i in n_grams:
            words += self.create_n_grams(clause, n_grams=i)
        return words