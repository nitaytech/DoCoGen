import argparse
import json
import os
from collections import defaultdict, Counter
from string import punctuation
from typing import List, Tuple, Dict, Union

import nltk
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_metric
from tqdm import tqdm
from transformers import (set_seed, PreTrainedTokenizerBase, AutoTokenizer, AutoModelForSequenceClassification,
                          T5ForConditionalGeneration, TrainingArguments, Trainer, pipeline, DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer, Pipeline)

# constants
MODEL_NAME = 't5-base'
CLASSIFIER_NAME = 'distilroberta-base'
BATCH_SIZE_DOCOGEN = 32
BATCH_SIZE_CLASSIFIER = 64
DOCOGEN_EPOCHS = 5
EVAL_SIZE = 1024
LABELED_SIZE = 16
SMOOTHING = [1, 5, 7]
MAX_N_GRAM = len(SMOOTHING)
MIN_N_OCCURRENCES = 10
N_ORIENTATIONS_PER_DOMAIN = 4
MASK_THRESHOLD = 0.08
MAX_LENGTH = 96
NUM_BEAMS = 4
UNKNOWN_VALUE = 'Unknown'
SEED = 42


def clean_text(text: str) -> str:
    # pad punctuations with spaces and remove double spaces
    puncs = set(punctuation).union({'\n', '\t'})
    puncs_trans = str.maketrans({k: f' {k} ' for k in puncs})
    return ' '.join([c for c in text.translate(puncs_trans).split(' ') if c != '']).strip()


def extract_n_grams(words: List[str], n: int) -> List[str]:
    return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]


def extract_n_grams_with_indices(words: List[str], n: int) -> List[Tuple[str, int, int]]:
    return [(' '.join(words[i:i + n]), i, i + n) for i in range(len(words) - n + 1)]


class Stemmer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        try:
            self.stemmer = nltk.SnowballStemmer("english", ignore_stopwords=True)
        except LookupError as e:
            nltk.download('stopwords')
            self.stemmer = nltk.SnowballStemmer("english", ignore_stopwords=True)
        # stemmed_vocab is used to store the non-stemmed value for each token in the vocabulary
        # (used to init orientations)
        stemmed_vocab = defaultdict(list)
        space_token = tokenizer.convert_ids_to_tokens(tokenizer('the')['input_ids'][0]).replace('the', '')
        for k, v in tokenizer.vocab.items():
            if not k.startswith(space_token):
                continue
            k = k.replace(space_token, '')
            stemmed_vocab[self.stemmer.stem(k).lower()].append(k)
        stemmed_vocab = {k: sorted(v, key=lambda w: (len(w), 0 if w.lower() == w else 1))[0]
                         for k, v in stemmed_vocab.items()}
        self.stemmed_vocab = stemmed_vocab

    def __call__(self, word: str):
        grams = []
        for w in word.lower().strip().split():
            w = self.stemmer.stem(w)
            if w.endswith('ies'):
                w = w[:-3] + 'y'
            elif w.endswith('ves'):
                w = w[:-3] + 'f'
            w = self.stemmed_vocab.get(w, w)
            grams.append(w)
        return ' '.join(grams)


class Masker:
    def __init__(self, tokenizer_name: str,
                 masking_scores: Dict[str, Dict[str, float]],
                 orientations: Dict[str, Dict[str, Union[str, int]]],
                 max_n_gram: int = MAX_N_GRAM,
                 mask_threshold: float = MASK_THRESHOLD):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.stemmer = Stemmer(self.tokenizer)
        self.masking_scores = masking_scores
        self.orientations = orientations
        self.domains = sorted(list(set([v['domain'] for v in self.orientations.values()])))
        self.domain2idx = {d: i for i, d in enumerate(self.domains)}
        self.max_n_gram = max_n_gram
        self.mask_threshold = mask_threshold

    def save_as_json(self, json_file_path: str):
        kwargs = {
            'tokenizer_name': self.tokenizer_name,
            'masking_scores': self.masking_scores,
            'orientations': self.orientations,
            'max_n_gram': self.max_n_gram,
            'mask_threshold': self.mask_threshold
        }
        with open(json_file_path, 'w') as f:
            json.dump(kwargs, f)

    @classmethod
    def load_masker(cls, json_file_path: str):
        with open(json_file_path, 'r') as f:
            kwargs = json.load(f)
        return cls(**kwargs)

    def masking_score(self, n_gram: str, origin_domain: str = None, destination_domain: str = None) -> float:
        n_gram = self.stemmer(n_gram)
        origin_domain = None if origin_domain not in self.domains else origin_domain
        destination_domain = None if destination_domain not in self.domains else destination_domain
        if (origin_domain is None and destination_domain is None) or n_gram not in self.masking_scores:
            return 0.0
        rho_orig = (self.masking_scores[n_gram][origin_domain] if origin_domain is not None
                    else max(self.masking_scores[n_gram].values()))
        rho_dest = (self.masking_scores[n_gram][destination_domain] if destination_domain is not None
                    else min(self.masking_scores[n_gram].values()))
        return rho_orig - rho_dest

    def mask_text(self, text: str, origin_domain: str = None, destination_domain: str = None) -> str:
        words = [w for w in clean_text(text).split()]
        for n in range(1, self.max_n_gram + 1):
            n_grams = extract_n_grams_with_indices(words, n)
            for n_gram, start_idx, end_idx in n_grams:
                if '<mask>' in n_gram:
                    continue
                if self.masking_score(n_gram, origin_domain, destination_domain) > self.mask_threshold:
                    for idx in range(start_idx, end_idx):
                        words[idx] = '<mask>'
        # merge consecutive masks and add a cnt
        cnt = 0
        masked_text = []
        for w in words:
            # skip if it is a mask and the previous word is also a mask
            if len(masked_text) > 0 and w == '<mask>' and masked_text[-1].startswith('<extra_id_'):
                continue
            # add a mask and increment the cnt
            elif w == '<mask>':
                masked_text.append(f'<extra_id_{cnt}>')
                cnt += 1
            else:
                masked_text.append(w)
        return ' '.join(masked_text)

    def prepare_text_for_train(self, text: str, origin_domain: str) -> Dict[str, str]:
        text = clean_text(text)
        stemmed_words = [self.stemmer(w) for w in text.split()]
        possible_orientations = [v['special_token'] for o, v in self.orientations.items()
                                 if v['domain'] == origin_domain and (o in stemmed_words or o == origin_domain)]
        orientation = np.random.choice(possible_orientations, 1)[0]
        return {'text': orientation + ' ' + self.mask_text(text, origin_domain, destination_domain=None),
                'labels': text}

    def prepare_text_for_inference(self, text: str, origin_domain: str,
                                   destination_domain: str, orientation: str = None) -> Dict[str, str]:
        text = clean_text(text)
        orientation = orientation if orientation is not None else destination_domain
        assert orientation in self.orientations and self.orientations[orientation]['domain'] == destination_domain
        orientation = self.orientations[orientation]['special_token']
        return {'text': orientation + ' ' + self.mask_text(text, origin_domain, destination_domain)}


def count_n_grams_for_domains(train_dataset: Dataset, stemmer: Stemmer,
                              max_n_gram: int = MAX_N_GRAM,
                              min_n_occurrences: int = MIN_N_OCCURRENCES) -> Dict[str, Dict[str, int]]:
    occurrences = {}
    domains = set(train_dataset['domain'])
    for domain, text in tqdm(zip(train_dataset['domain'], train_dataset['text'])):
        if domain is None or domain == UNKNOWN_VALUE:
            continue
        words = [stemmer(w) for w in clean_text(text).split()]
        for n in range(1, max_n_gram + 1):
            n_grams = set(extract_n_grams(words, n))
            for n_gram in n_grams:
                if n_gram not in occurrences:
                    occurrences[n_gram] = {d: 0 for d in domains}
                occurrences[n_gram][domain] += 1
    # filter n_grams with less than min_n_occurrences
    occurrences = {n_gram: n_gram_occs for n_gram, n_gram_occs in occurrences.items()
                   if sum(n_gram_occs.values()) >= min_n_occurrences}
    return occurrences


def prepare_orientations(rep_words: Dict[str, List[Tuple[float, str]]],
                         tokenizer: PreTrainedTokenizerBase,
                         n_orientations_per_domain: int = N_ORIENTATIONS_PER_DOMAIN):
    space_token = tokenizer.convert_ids_to_tokens(tokenizer('the')['input_ids'][0]).replace('the', '')
    orientations = {}
    # find orientations which are the most representative words of each domain, including the domain name
    for d, rep_words_d in rep_words.items():
        top_rep_words_t = sorted(rep_words_d, key=lambda x: x[0], reverse=True)[:n_orientations_per_domain]
        top_rep_words_t = [w for _, w in top_rep_words_t]
        if d in [w for _, w in rep_words_d]:
            top_rep_words_t = [d] + [w for w in top_rep_words_t if w != d]
            top_rep_words_t = top_rep_words_t[:n_orientations_per_domain]
            orientations.update({o: {'domain': d, 'init_id': tokenizer.vocab[space_token + o]}
                                 for o in top_rep_words_t})
        else:
            orientations[d] = {'domain': d, 'init_id': tokenizer.vocab[space_token + top_rep_words_t[-1]]}
            orientations.update({o: {'domain': d, 'init_id': tokenizer.vocab[space_token + o]}
                                 for o in top_rep_words_t[:-1]})
    # add orientations to tokenizer as special tokens - in the original paper we didn't do so.
    # We implemented a new embedding matrix for the orientations.
    # However, in order to make the code more simple and to support HuggingFace Trainer,
    # we use this alternative ("hacking") way - as the special_tokens have learnable embeddings and are not in use.
    cnt = 99
    for o in sorted(list(orientations.keys())):
        orientations[o]['special_token'] = f'<extra_id_{cnt}>'
        orientations[o]['special_id'] = tokenizer.vocab[f'<extra_id_{cnt}>']
        cnt -= 1
    return orientations


def prepare_masker(dataset: Dataset,
                   tokenizer: PreTrainedTokenizerBase,
                   max_n_gram: int = MAX_N_GRAM,
                   min_n_occurrences: int = MIN_N_OCCURRENCES,
                   smoothing: List[float] = SMOOTHING,
                   mask_threshold: float = MASK_THRESHOLD):
    tokenizer_name = tokenizer.name_or_path
    domains = set(dataset['domain'])
    stemmer = Stemmer(tokenizer)
    occurrences = count_n_grams_for_domains(dataset, stemmer, max_n_gram, min_n_occurrences)
    masking_scores = {}
    rep_words = {d: [] for d in domains}
    tokens = set(stemmer.stemmed_vocab.values())
    domain_occurrences = Counter(dataset['domain'])
    for n_gram in occurrences:
        total = sum(occurrences[n_gram].values())
        smoothing_factor = smoothing[len(n_gram.split()) - 1] / len(domains)
        # tP(D|W) is *proportional* to the number of times w appears in D divided by D size
        # this is because we assume a uniform prior over D and by using bayes we get:
        # P(D|W) = P(W|D) * P(D) / P(W) proportional to P(W|D) = (#(w in D) + smoothing_factor) / D size
        probabilities = {d: (v + smoothing_factor) / domain_occurrences[d] for d, v in occurrences[n_gram].items()}
        # normalizing the probabilities to sum to 1 (since they are proportional, and we need to fix it)
        probabilities = {d: v / sum(probabilities.values()) for d, v in probabilities.items()}
        entropy = sum([-p * np.log(p) for p in probabilities.values()])
        neg_entropy = 1.0 - entropy / np.log(len(probabilities))
        masking_scores[n_gram] = {}
        for d, p in probabilities.items():
            rho = p * neg_entropy
            masking_scores[n_gram][d] = rho
            if n_gram in tokens:
                rep_words[d].append((np.log(total) * rho, n_gram))
    orientations = prepare_orientations(rep_words, tokenizer)
    return Masker(tokenizer_name, masking_scores, orientations, max_n_gram, mask_threshold)


def load_datasets(dataset_file_path: str,
                  tokenizer: PreTrainedTokenizerBase,
                  domains_to_control: List[str],
                  max_length: int = MAX_LENGTH,
                  eval_size: int = EVAL_SIZE,
                  labeled_size: int = LABELED_SIZE):

    def preprocess_dataset_function(example: Dict) -> Dict:
        # fix examples from airline domain by removing the first sentence (which describe the flight number)
        # and truncate the examples to max_length
        text = example['text']
        domain = example['domain']
        if domain == 'airline':
            splitted = text.split('.')
            if len(splitted) > 1 and len(splitted[1]) > 0:
                splitted = splitted[1:]
            text = '.'.join(splitted).strip()
        text = clean_text(text)
        return {'text': tokenizer.batch_decode([tokenizer(text, max_length=max_length, truncation=True)['input_ids']],
                                               skip_special_tokens=True)[0]}

    with open(dataset_file_path, 'r') as f:
        dataset = Dataset.from_dict(json.load(f))
    dataset = dataset.filter(lambda example: example['domain'] in domains_to_control)
    dataset = dataset.map(preprocess_dataset_function, batched=False)
    unlabeled_dataset = dataset.filter(lambda example: example['split'] == 'unlabeled')
    total_examples = len(unlabeled_dataset)
    assert eval_size < total_examples
    train_indices = np.random.choice(list(range(total_examples)), total_examples - eval_size, replace=False)
    train_dataset = unlabeled_dataset.select(train_indices)
    eval_dataset = unlabeled_dataset.select([i for i in range(total_examples) if i not in train_indices])
    labeled_dataset = dataset.filter(lambda example: example['split'] in ['train', 'validation'])
    if labeled_size is not None:
        labeled_indices = np.random.choice(list(range(len(labeled_dataset))), labeled_size, replace=False)
        labeled_dataset = labeled_dataset.select(labeled_indices)
    return train_dataset, eval_dataset, labeled_dataset


def prepare_preprocess_function_docogen_train(masker: Masker, max_length: int = MAX_LENGTH):

    def preprocess_function_docogen_train(example: Dict) -> Dict:
        text = clean_text(example['text'])
        origin_domain = example['domain']
        masked_text = masker.prepare_text_for_train(text, origin_domain)
        masked_text, labels = masked_text['text'], masked_text['labels']
        model_inputs = masker.tokenizer(masked_text, max_length=max_length, truncation=True)
        # Setup the tokenizer for targets
        with masker.tokenizer.as_target_tokenizer():
            labels = masker.tokenizer(labels, max_length=max_length, truncation=True)['input_ids']
        return {'input_ids': model_inputs['input_ids'],
                'attention_mask': model_inputs['attention_mask'],
                'labels': labels}

    return preprocess_function_docogen_train


def prepare_preprocess_function_docogen_inference(masker: Masker, max_length: int = MAX_LENGTH):

    def preprocess_function_docogen_inference(example: Dict) -> Dict:
        text = clean_text(example['text'])
        origin_domain = example['domain']
        destination_domain = np.random.choice(list(masker.domains), 1)[0]
        domain_orientations = [o for o, v in masker.orientations.items() if v['domain'] == destination_domain]
        orientation = np.random.choice(domain_orientations, 1)[0]
        masked_text = masker.prepare_text_for_inference(text, origin_domain, destination_domain, orientation)
        masked_text = masked_text['text']
        model_inputs = masker.tokenizer(masked_text, max_length=max_length, truncation=True)
        label = [masker.domain2idx[destination_domain]]
        return {'input_ids': model_inputs['input_ids'],
                'attention_mask': model_inputs['attention_mask'],
                'labels': label}

    return preprocess_function_docogen_inference


def prepare_preprocess_function_classifier(masker: Masker, classifier_tokenizer: PreTrainedTokenizerBase,
                                           max_length: int = MAX_LENGTH):

    def preprocess_function_classifier(example: Dict) -> Dict:
        text = clean_text(example['text'])
        label = masker.domain2idx[example['domain']]
        model_inputs = classifier_tokenizer(text, max_length=max_length, truncation=True)
        return {'input_ids': model_inputs['input_ids'],
                'attention_mask': model_inputs['attention_mask'],
                'labels': label}

    return preprocess_function_classifier


def prepare_datasets_and_masker(dataset_file_path: str,
                                output_dir: str,
                                domains_to_control: List[str],
                                model_name: str = MODEL_NAME,
                                classifier_name: str = CLASSIFIER_NAME,
                                max_length: int = MAX_LENGTH,
                                eval_size: int = EVAL_SIZE,
                                labeled_size: int = LABELED_SIZE,
                                max_n_gram: int = MAX_N_GRAM,
                                min_n_occurrences: int = MIN_N_OCCURRENCES,
                                smoothing: List[float] = SMOOTHING,
                                mask_threshold: float = MASK_THRESHOLD):
    # prepare datasets and masker for training the docogen
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_name)
    train_dataset, eval_dataset, labeled_dataset = load_datasets(dataset_file_path, tokenizer, domains_to_control,
                                                                 max_length, eval_size, labeled_size)
    masker = prepare_masker(train_dataset, tokenizer, max_n_gram, min_n_occurrences, smoothing, mask_threshold)

    preprocess_classifier = prepare_preprocess_function_classifier(masker, classifier_tokenizer, max_length)
    preprocess_docogen_train = prepare_preprocess_function_docogen_train(masker, max_length)
    preprocess_docogen_inference = prepare_preprocess_function_docogen_inference(masker, max_length)
    columns = [c for c in train_dataset.column_names if c not in ['input_ids', 'attention_mask', 'labels']]

    train_dataset_for_classifier = train_dataset.map(preprocess_classifier, batched=False, remove_columns=columns)
    eval_dataset_for_classifier = eval_dataset.map(preprocess_classifier, batched=False, remove_columns=columns)
    train_dataset_for_docogen = train_dataset.map(preprocess_docogen_train, batched=False, remove_columns=columns)
    eval_dataset_for_docogen = eval_dataset.map(preprocess_docogen_inference, batched=False, remove_columns=columns)

    # save
    os.makedirs(output_dir, exist_ok=True)
    masker.save_as_json(os.path.join(output_dir, 'masker.json'))
    train_dataset_for_classifier.save_to_disk(os.path.join(output_dir, 'train_dataset_for_classifier'))
    eval_dataset_for_classifier.save_to_disk(os.path.join(output_dir, 'eval_dataset_for_classifier'))
    train_dataset_for_docogen.save_to_disk(os.path.join(output_dir, 'train_dataset_for_docogen'))
    eval_dataset_for_docogen.save_to_disk(os.path.join(output_dir, 'eval_dataset_for_docogen'))
    labeled_dataset.save_to_disk(os.path.join(output_dir, 'labeled_dataset'))


def train_domain_classifier(output_dir: str,
                            classifier_name: str = CLASSIFIER_NAME,
                            batch_size_classifier: int = BATCH_SIZE_CLASSIFIER):
    # train domain classifier which is used for the evaluation step of DoCoGen.
    # we use it ti predict the domain of the generated counterfactual and calculate the accuracy of the model.

    # load datasets and masker
    train_dataset_for_classifier = Dataset.load_from_disk(os.path.join(output_dir, 'train_dataset_for_classifier'))
    eval_dataset_for_classifier = Dataset.load_from_disk(os.path.join(output_dir, 'eval_dataset_for_classifier'))
    masker = Masker.load_masker(os.path.join(output_dir, 'masker.json'))

    # load classifier
    classifier = AutoModelForSequenceClassification.from_pretrained(classifier_name, num_labels=len(masker.domains))
    classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_name)
    accuracy_metric = load_metric('accuracy')

    def compute_accuracy(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    classifier_output_dir = os.path.join(output_dir, 'domain_classifier_training')
    training_args = TrainingArguments(output_dir=classifier_output_dir,
                                      evaluation_strategy='epoch',
                                      save_total_limit=1,
                                      save_strategy='epoch',
                                      metric_for_best_model='accuracy',
                                      greater_is_better=True,
                                      num_train_epochs=1,
                                      per_device_train_batch_size=batch_size_classifier,
                                      per_device_eval_batch_size=batch_size_classifier * 2,
                                      gradient_accumulation_steps=1,
                                      learning_rate=5e-5,
                                      weight_decay=1e-5,
                                      adam_epsilon=1e-8,
                                      label_smoothing_factor=0.2,
                                      report_to='none')
    trainer = Trainer(model=classifier,
                      args=training_args,
                      train_dataset=train_dataset_for_classifier,
                      eval_dataset=eval_dataset_for_classifier,
                      tokenizer=classifier_tokenizer,
                      compute_metrics=compute_accuracy)
    trainer.train()
    scores = trainer.evaluate()
    for k, v in scores.items():
        print(f'{k}: {v:.2f}')
    trainer.save_model(os.path.join(output_dir, 'trained_domain_classifier'))


def train_docogen(output_dir: str,
                  model_name: str = MODEL_NAME,
                  batch_size_docogen: int = BATCH_SIZE_DOCOGEN,
                  docogen_epochs: int = DOCOGEN_EPOCHS,
                  max_length: int = MAX_LENGTH,
                  num_beams: int = NUM_BEAMS):
    # load datasets and masker
    train_dataset_for_docogen = Dataset.load_from_disk(os.path.join(output_dir, 'train_dataset_for_docogen'))
    eval_dataset_for_docogen = Dataset.load_from_disk(os.path.join(output_dir, 'eval_dataset_for_docogen'))
    masker = Masker.load_masker(os.path.join(output_dir, 'masker.json'))
    tokenizer = masker.tokenizer

    # load domain classifier
    classifier = AutoModelForSequenceClassification.from_pretrained(os.path.join(output_dir,
                                                                                 'trained_domain_classifier'))
    classifier_tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_dir, 'trained_domain_classifier'))
    classification_pipeline = pipeline('text-classification', model=classifier, tokenizer=classifier_tokenizer)
    accuracy_metric = load_metric('accuracy')

    docogen_output_dir = os.path.join(output_dir, 'docogen_training')
    epoch = [0]  # to be used as a mutable-global variable

    def compute_accuracy_for_docogen(eval_pred):
        generation_ids, labels = eval_pred
        labels = labels[:, 0]
        generations = tokenizer.batch_decode(generation_ids,
                                             skip_special_tokens=True, clean_up_tokenization_spaces=True)
        predictions = np.array([int(classification_pipeline(gen)[0]['label'].split('_')[-1]) for gen in generations])
        dlabels = [masker.domains[label] for label in labels]
        dpredictions = [masker.domains[pred] for pred in predictions]
        df = pd.DataFrame(data={'generations': generations, 'labels': dlabels, 'predictions': dpredictions})
        df.to_csv(os.path.join(docogen_output_dir, f'evaluation_e{epoch[0]}.csv'), index=False)
        epoch[0] += 1
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # load DoCoGen and initialize the orientations
    docogen = T5ForConditionalGeneration.from_pretrained(model_name)
    with torch.no_grad():
        for orientation, values in masker.orientations.items():
            init_id, special_id = values['init_id'], values['special_id']
            init_w = torch.tensor(docogen.shared.weight[init_id], device=docogen.shared.weight.device)
            docogen.shared.weight[special_id] = init_w

    eval_batch_size = int(2 * batch_size_docogen / num_beams)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=docogen, padding=True, return_tensors='pt')
    training_args = Seq2SeqTrainingArguments(output_dir=docogen_output_dir,
                                             evaluation_strategy='epoch',
                                             save_total_limit=1,
                                             save_strategy='epoch',
                                             metric_for_best_model='accuracy',
                                             greater_is_better=True,
                                             num_train_epochs=docogen_epochs,
                                             per_device_train_batch_size=batch_size_docogen,
                                             per_device_eval_batch_size=eval_batch_size,
                                             gradient_accumulation_steps=1,
                                             learning_rate=5e-5,
                                             weight_decay=1e-5,
                                             adam_epsilon=1e-8,
                                             predict_with_generate=True,
                                             generation_max_length=max_length,
                                             generation_num_beams=num_beams,
                                             report_to='none')
    trainer = Seq2SeqTrainer(model=docogen,
                             args=training_args,
                             train_dataset=train_dataset_for_docogen,
                             eval_dataset=eval_dataset_for_docogen,
                             data_collator=data_collator,
                             tokenizer=tokenizer,
                             compute_metrics=compute_accuracy_for_docogen)
    trainer.train()
    scores = trainer.evaluate()
    for k, v in scores.items():
        print(f'{k}: {v:.2f}')
    trainer.save_model(os.path.join(output_dir, 'trained_docogen'))


def generate(masker: Masker,
             generation_pipeline: Pipeline,
             text: str,
             origin_domain: str,
             destination_domain: str,
             orientation: str = None,
             **generate_kwargs) -> str:
    masked_text = masker.prepare_text_for_inference(text, origin_domain, destination_domain, orientation)['text']
    generated_output = generation_pipeline(masked_text, **generate_kwargs)[0]
    # the output depends on the HF's version
    if 'generated_token_ids' in generated_output:
        generated_text = masker.tokenizer.batch_decode(generated_output['generated_token_ids'],
                                                       skip_special_tokens=True, clean_up_tokenization_spaces=True)
    else:
        generated_text = generated_output['generated_text']
    return generated_text


def print_generated_text(text: str,
                         origin_domain: str,
                         destination_domain: str,
                         generated_text: str,
                         orientation: str = None):
    orientation = "(" + orientation + ")" if orientation is not None else ""
    print(f'Original, {origin_domain}:')
    print(text)
    print()
    print(f'DoCoGen, {origin_domain} -> {destination_domain}{orientation}:')
    print(generated_text)
    print(f"{'-' * 50}")


def generate_domain_counterfactuals(output_dir: str,
                                    max_length: int = MAX_LENGTH,
                                    num_beams: int = NUM_BEAMS,
                                    generate_all_orientations: bool = False,
                                    print_generated_texts: bool = True):
    # load dataset and masker
    labeled_dataset = Dataset.load_from_disk(os.path.join(output_dir, 'labeled_dataset'))
    masker = Masker.load_masker(os.path.join(output_dir, 'masker.json'))

    # load DoCoGen
    docogen_path = os.path.join(output_dir, 'trained_docogen')
    tokenizer = AutoTokenizer.from_pretrained(docogen_path)
    docogen = T5ForConditionalGeneration.from_pretrained(docogen_path)

    # prepare generation pipeline and kwargs
    generation_pipeline = pipeline('text2text-generation', model=docogen, tokenizer=tokenizer)
    generate_kwargs = dict(num_beams=num_beams, num_beam_groups=1, do_sample=False, diversity_penalty=0.0,
                           temperature=1.0, max_length=max_length)

    # truncate texts to max_length
    texts = tokenizer.batch_decode(tokenizer(labeled_dataset['text'], max_length=max_length,
                                             truncation=True)['input_ids'], skip_special_tokens=True)
    text_domains = labeled_dataset['domain']
    outputs = []
    for text, origin_domain in zip(texts, text_domains):
        destination_domains = [v['domain'] for v in masker.orientations.values() if v['domain'] != origin_domain]
        orientations = [o for o, v in masker.orientations.items() if v['domain'] != origin_domain]
        if not generate_all_orientations:
            destination_domains = [np.random.choice(destination_domains, 1)[0]]
            orientations = destination_domains
        for destination_domain, orientation in zip(destination_domains, orientations):
            generated_text = generate(masker, generation_pipeline, text,
                                      origin_domain, destination_domain, orientation, **generate_kwargs)
            if print_generated_texts:
                print_generated_text(text, origin_domain, destination_domain, generated_text, orientation)
            outputs.append({'text': text, 'origin_domain': origin_domain, 'destination_domain': destination_domain,
                            'orientation': orientation, 'generated_text': generated_text})
    df = pd.DataFrame(outputs)
    df.to_csv(os.path.join(output_dir, 'domain_counterfactuals.csv'), index=False)


def main(args):
    dataset_file_path = args.dataset_file_path
    output_dir = args.output_dir
    domains_to_control = args.domains_to_control
    model_name = args.model_name
    classifier_name = args.classifier_name
    max_length = args.max_length
    eval_size = args.eval_size
    labeled_size = args.labeled_size
    min_n_occurrences = args.min_n_occurrences
    smoothing = args.smoothing
    batch_size_classifier = args.batch_size_classifier
    batch_size_docogen = args.batch_size_docogen
    docogen_epochs = args.docogen_epochs
    num_beams = args.num_beams
    generate_all_orientations = args.generate_all_orientations
    print_generated_texts = args.print_generated_texts
    max_n_gram = len(smoothing)

    seed = args.seed
    set_seed(seed)

    prepare_datasets_and_masker(dataset_file_path, output_dir, domains_to_control, model_name, classifier_name,
                                max_length, eval_size, labeled_size, max_n_gram, min_n_occurrences, smoothing)
    train_domain_classifier(output_dir, classifier_name, batch_size_classifier)
    train_docogen(output_dir, model_name, batch_size_docogen, docogen_epochs, max_length, num_beams)
    generate_domain_counterfactuals(output_dir, max_length, num_beams,
                                    generate_all_orientations, print_generated_texts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file_path', type=str,
                        help='(str) Path to the dataset file (.json). Must include the following fields:'
                             ' "text", "domain"')
    parser.add_argument('--output_dir', type=str,
                        help='(str) Path to the output directory where the following files will be save: '
                             '(1) preprocessed datasets, (2) masker, (3) trained domain classifier, '
                             '(4) trained DoCoGen (including the training outputs), (5) domain-counterfactuals.')
    parser.add_argument('--domains_to_control', type=str, nargs='+',
                        help='(list of str) List of domains to control (should be values of the "domain" field).')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help='(str) Name of the T5 model to use for the DoCoGen.')
    parser.add_argument('--classifier_name', type=str, default=CLASSIFIER_NAME,
                        help='(str) Name of the pre-trained model which will be trained to be a domain classifier.'
                             ' This model is part of the evaluation step of DoCoGen.')
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH,
                        help='(int) Maximum length of the input and the generated texts.')
    parser.add_argument('--eval_size', type=int, default=EVAL_SIZE,
                        help='(int) Size of the evaluation set. Use a small value, since the evaluation step of '
                             'DoCoGen includes generations -- and it is really slow compared to the training.')
    parser.add_argument('--labeled_size', type=int, default=LABELED_SIZE,
                        help='(int) Size of the labeled dataset which will be used to generate domain-counterfactuals.'
                             ' Use a small size or use `None` if you are interested in the whole dataset.')
    parser.add_argument('--min_n_occurrences', type=int, default=MIN_N_OCCURRENCES,
                        help='(int) An n-gram which occurs less than this value will have a masking score of zero.')
    parser.add_argument('--smoothing', type=float, nargs='+', default=SMOOTHING,
                        help='(list of ints/floats) The n-th element is the smoothing hyperparameter for '
                             'an n sized n-gram. These hyperparameters are used to smooth the masking score'
                             ' (higher values give more weight to the uniform prior). In addition, the length of the '
                             'smoothing list determines the maximum n-gram size.')
    parser.add_argument('--batch_size_classifier', type=int, default=BATCH_SIZE_CLASSIFIER,
                        help='(int) Batch size for the domain classifier.')
    parser.add_argument('--batch_size_docogen', type=int, default=BATCH_SIZE_DOCOGEN,
                        help='(int) Batch size for DoCoGen.')
    parser.add_argument('--docogen_epochs', type=int, default=DOCOGEN_EPOCHS,
                        help='(int) Number of epochs for DoCoGen training.')
    parser.add_argument('--num_beams', type=int, default=NUM_BEAMS,
                        help='(int) Number of beams for DoCoGen generation.')
    parser.add_argument('--generate_all_orientations', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='(bool) If `True`, all possible orientations will be generated for each example.'
                             ' Otherwise, randomly sample an orientation.')
    parser.add_argument('--print_generated_texts', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='(bool) If `True`, the generated texts will be printed.')
    parser.add_argument('--seed', type=int, default=SEED)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()
    main(args)
