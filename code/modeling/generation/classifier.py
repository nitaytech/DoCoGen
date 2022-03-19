import logging
from argparse import Namespace
from pathlib import Path
from typing import List, Any, Dict, Union

import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, LongTensor, FloatTensor
from torch.nn import Module, Linear, ModuleDict
from transformers import (PreTrainedModel, AutoModelForPreTraining, AutoTokenizer,
                          AdamW, get_linear_schedule_with_warmup)
from transformers.file_utils import ModelOutput

from modeling.masking.mapper import ConceptsMapping
from project_utils.constants import DIM, UNK, T5_MODEL_NAME
from project_utils.functions import concat_dicts, subset_dict
from modeling.generation.modules import LabelSmoothingLoss, GradReverseLayerFunction


class EncoderClassifier(Module):
    def __init__(self,
                 encoder_model: PreTrainedModel,
                 concepts_mapping: Union[str, Path, ConceptsMapping, Dict[str, List[str]]],
                 concepts_to_predict: List[str] = None,
                 concept_fc_dict: Dict[str, Module] = None,
                 label_smoothing: float = 0.0,
                 ignore_unknown: bool = True):
        """
        The EncoderClassifier is a torch.Module class consisting a PreTrainedModel encoder (which is used to embed a
        sentence by taking the average of the hidden states of its tokens) and a dictionary of Linear layers
         (which are used to predict values of concepts, for example, a concept can be 'domain' and its values
          can be [UNK, 'airline', 'books', 'kitchen']. The UNK is the Unknown value (None/NULL) and can be ignored.
        :param encoder_model: a PreTrainedModel transformer (see: https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/model#transformers.PreTrainedModel)
        :param concepts_mapping: a ConceptsMapping object, or a path-like of a json containing the
         ConceptsMapping object, or a dict where the keys are concepts and the values are lists of the possible
          values of each concept.
        :param concepts_to_predict: a list of strings, each string is a concept you wish to predict by the classifier.
        :param concept_fc_dict: [Optional] a dict where the keys are concepts and the values are Linear layers
         used for predicting the concepts.
        :param label_smoothing: a float for label smoothing (0.0 for no smoothing).
        :param ignore_unknown: a bool. If False the UNK value is a possible prediction of the model, otherwise this
         value is ignored.
        """
        super().__init__()
        try:
            model_dim = encoder_model.config.d_model
        except:
            model_dim = DIM
        if concepts_to_predict is not None:
            concepts_to_predict = concepts_to_predict.copy()
        else:
            concepts_to_predict = concepts_mapping.concepts.copy()
        concept_fc_dict = concept_fc_dict if concept_fc_dict is not None else {}
        self.concepts_mapping = ConceptsMapping.from_multiple_types(concepts_mapping)
        self.ignore_unknown = ignore_unknown
        for concept in concept_fc_dict:
            if concept not in self.concepts_mapping.concepts or concept not in concepts_to_predict:
                concept_fc_dict[concept].pop(concept)
        for concept, concept_values in self.concepts_mapping.concepts_values.items():
            if concept not in concepts_to_predict:
                if concept in concept_fc_dict:
                    concept_fc_dict.pop(concept)
                continue
            elif concept not in concept_fc_dict:
                n_classes = len(self.concepts_mapping.concept_values(concept))
                if self.ignore_unknown:
                    n_classes -= 1
                concept_fc_dict[concept] = Linear(model_dim, n_classes)

        self.encoder_model = encoder_model
        self.concept_fc_dict = ModuleDict(concept_fc_dict)
        self.ls_loss = LabelSmoothingLoss(label_smoothing)

    @property
    def concepts(self):
        return list(self.concept_fc_dict.keys())

    def get_modules(self) -> Dict[str, Any]:
        return_dict = {'encoder_model': self.encoder_model,
                       'concept_fc_dict': self.concept_fc_dict}
        return return_dict

    def set_modules(self,
                    encoder_model: PreTrainedModel = None,
                    concept_fc_dict: ModuleDict = None,
                    **kwargs):
        if encoder_model is not None:
            self.encoder_model = encoder_model
        if concept_fc_dict is not None:
            self.concept_fc_dict = concept_fc_dict

    def encoder_model_forward(self,
                              input_ids: LongTensor,
                              attention_mask: LongTensor,
                              with_encoder_grad: bool = False,
                              **encoder_kwargs):
        if not with_encoder_grad:
            with torch.no_grad():
                encoder_outputs: ModelOutput = self.encoder_model(input_ids=input_ids,
                                                                  attention_mask=attention_mask,
                                                                  **encoder_kwargs)
        else:
            encoder_outputs: ModelOutput = self.encoder_model(input_ids=input_ids,
                                                              attention_mask=attention_mask,
                                                              **encoder_kwargs)
        return encoder_outputs

    def _extract_hidden_states(self,
                               input_ids: LongTensor,
                               attention_mask: LongTensor,
                               with_encoder_grad: bool = False,
                               **encoder_kwargs):
        encoder_outputs = self.encoder_model_forward(input_ids=input_ids,
                                                     attention_mask=attention_mask,
                                                     with_encoder_grad=with_encoder_grad,
                                                     **encoder_kwargs)
        hidden_states = encoder_outputs[0]
        return hidden_states

    def encode(self,
               input_ids: LongTensor,
               attention_mask: LongTensor = None,
               with_encoder_grad: bool = False,
               **encoder_kwargs):
        """
        Encode a sentence by first extracting the hidden states of each token, and then averaging them,
         ignoring pad tokens.
        :param input_ids:
        :param attention_mask:
        :param with_encoder_grad:
        :param encoder_kwargs:
        :return: Tensor
        """
        hidden_states = self._extract_hidden_states(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    with_encoder_grad=with_encoder_grad,
                                                    **encoder_kwargs)
        # mean hidden state, ignoring tokens with 0 attention mask
        attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        representation = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        return representation

    def predict_token_scores(self,
                             input_ids: LongTensor,
                             attention_mask: LongTensor,
                             with_encoder_grad: bool = False) -> Dict[str, Any]:
        hidden_states = self._extract_hidden_states(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    with_encoder_grad=with_encoder_grad)
        token_scores = {}
        for concept, concept_classifier in self.concept_fc_dict.items():
            token_scores[concept] = concept_classifier(hidden_states)
        return token_scores

    def predict_scores(self,
                       input_ids: LongTensor,
                       attention_mask: LongTensor,
                       with_encoder_grad: bool = False) -> Dict[str, FloatTensor]:
        token_scores = self.predict_token_scores(input_ids=input_ids,
                                                 attention_mask=attention_mask,
                                                 with_encoder_grad=with_encoder_grad)
        scores = {}
        for concept, token_concept_scores in token_scores.items():
            # mean score, ignoring tokens with 0 attention mask
            mask = attention_mask.unsqueeze(-1).expand_as(token_concept_scores)
            scores[concept] = (token_concept_scores * mask).sum(dim=1) / mask.sum(dim=1)
        return scores

    def forward(self,
                input_ids: LongTensor,
                attention_mask: LongTensor,
                with_encoder_grad: bool = False) -> Dict[str, FloatTensor]:
        """
        :param input_ids:
        :param attention_mask:
        :param with_encoder_grad:
        :return: A dict where the keys are the concepts, and the values are Tensors of the predictions scores.
        """
        return self.predict_scores(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   with_encoder_grad=with_encoder_grad)

    def predict_concepts(self,
                         input_ids: LongTensor,
                         attention_mask: LongTensor) -> Dict[str, List[str]]:
        """

        :param input_ids:
        :param attention_mask:
        :return: A dict where the keys are the concepts, and the values are lists of strings. Entry i in each list
        corresponds to the prediction of example i.
        """
        scores = self.predict_scores(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     with_encoder_grad=False)
        predictions = {}
        for concept, concept_scores in scores.items():
            concept_predictions = torch.max(concept_scores, 1)[1].tolist()
            predictions[concept] = [self.concepts_mapping.index_to_value(concept, i, self.ignore_unknown)
                                         for i in concept_predictions]
        return predictions

    def scores_to_probabilities(self,
                                concepts_scores: Dict[str, Tensor],
                                return_dict: bool = False) -> Dict[str, Any]:
        concepts_probabilities = {}
        for concept in self.concepts:
            concept_scores = concepts_scores[concept]
            probs = torch.softmax(concept_scores, 1)
            if return_dict:
                probs = probs.detach().cpu().tolist()
                concepts_probabilities[concept] = []
                for example_probs in probs:
                    ps = {self.concepts_mapping.index_to_value(concept, i, self.ignore_unknown): p
                          for i, p in enumerate(example_probs)}
                    concepts_probabilities[concept].append(ps)
            else:
                concepts_probabilities[concept] = probs
        return concepts_probabilities

    def predict_probabilities(self, input_ids: LongTensor,
                              attention_mask: LongTensor,
                              return_dict: bool = False,
                              with_encoder_grad: bool = True) -> Dict[str, Any]:
        scores = self.predict_scores(input_ids, attention_mask, with_encoder_grad=with_encoder_grad)
        return self.scores_to_probabilities(scores, return_dict=return_dict)

    def predict_concepts_forward_step(self, input_ids: LongTensor,
                                      attention_mask: LongTensor,
                                      concepts_values: Dict[str, List[str]],
                                      ignore_unknown: bool = True,
                                      return_predictions: bool = False,
                                      return_probabilities: bool = False,
                                      return_scores: bool = False,
                                      with_encoder_grad: bool = True,
                                      **kwargs) -> Dict[str, Any]:
        """
        A full forward step used to train the encoder classifier.
        :param input_ids:
        :param attention_mask:
        :param concepts_values: a dict where the keys are the concepts and the values are lists of the TRUE values
         (these are used to calculate the loss and measure the performance)
        :param ignore_unknown: if True, ignores UNK values (not calculating/measuring their loss/performance).
        :param return_predictions: a bool, if True returns the predictions (stored at the 'predictions' key
         in the returned dict). The predictions are a dict where the keys are concepts and the values are lists of
         the predicted values.
        :param return_probabilities: a bool, if True returns the probabilities (stored at the 'probabilities' key
         in the returned dict). The probabilities are a dict where the keys are concepts and the values are lists of
         dicts, each dict is a mapping between a value and the probability assigned to it.
         For example: {'concept': [{'value1': p11, 'value2': p12}, {'value1': p21, 'value2': p22},...]}
        :param return_scores: a bool, if True returns the scores (value of the 'scores' key
         in the returned dict). The scores are a dict where the keys are concepts and the values are FloatTensors.
        :param with_encoder_grad: a bool. if False does not calculate gradients (with torch.no_grad())
        :param kwargs:
        :return: a dict (`loss_dict`) with the following keys:
            * 'loss' - for backward()
            * 'predict_loss' - mean over of all 'predict_loss_{concept}', equals to 'loss'.
            * 'predict_acc' - mean over of all 'predict_acc_{concept}'.
            * 'predict_loss_{concept}' - a loss calculated over a specific concept (for each concept).
            * 'predict_acc_{concept}' - the accuracy measured for a specific concept (for each concept).
            * 'predictions', 'probabilities', 'scores' - see above.
        """
        ignore_unknown = ignore_unknown if ignore_unknown is not None else self.ignore_unknown
        scores = self.predict_scores(input_ids, attention_mask, with_encoder_grad=with_encoder_grad)
        predictions, losses, accuracies = {}, {}, {}
        for concept in self.concepts:
            concept_labels = concepts_values[concept]
            concept_scores = scores[concept]
            concept_predictions = torch.max(concept_scores, 1)[1]
            n_classes = len(self.concepts_mapping.concept_values(concept))
            if self.ignore_unknown:
                n_classes -= 1
            predictions[concept] = [self.concepts_mapping.index_to_value(concept, i, self.ignore_unknown)
                                         for i in concept_predictions.tolist()]
            labels = torch.tensor([self.concepts_mapping.value_to_index(concept, l, self.ignore_unknown)
                                   for l in concept_labels],
                                  dtype=input_ids.dtype, device=input_ids.device)
            if ignore_unknown:
                not_none_mask = labels != self.concepts_mapping.value_to_index(concept, UNK, self.ignore_unknown)
                labels = labels[not_none_mask]
                concept_scores = concept_scores[not_none_mask]
                concept_predictions = concept_predictions[not_none_mask]
            if labels.shape[0] == 0:
                losses[concept] = torch.as_tensor(float('nan'))
                accuracies[concept] = torch.as_tensor(float('nan'))
            else:
                losses[concept] = self.ls_loss.forward(concept_scores, labels, n_classes=n_classes, dim=-1)
                accuracies[concept] = torch.sum(concept_predictions == labels) / concept_predictions.shape[0]
        notnan_losses = [cl for cl in losses.values() if not torch.isnan(cl).any().item()]
        if len(notnan_losses) > 0:
            loss = sum(notnan_losses) / max(1, len(notnan_losses))
        else:
            loss = torch.as_tensor(float('nan'))
        notnan_acces = [ca for ca in accuracies.values() if not torch.isnan(ca).any().item()]
        if len(notnan_acces) > 0:
            accuracy = sum(notnan_acces) / max(1, len(notnan_acces))
        else:
            accuracy = torch.as_tensor(float('nan'))
        loss_dict = {'loss': loss, 'predict_loss': loss, 'predict_acc': accuracy}
        loss_dict.update({f'predict_{k}_loss': v for k, v in losses.items()})
        loss_dict.update({f'predict_{k}_acc': v for k, v in accuracies.items()})
        if return_predictions:
            loss_dict['predictions'] = predictions
        if return_probabilities:
            loss_dict['probabilities'] = self.scores_to_probabilities(scores, return_dict=True)
        if return_scores:
            loss_dict['scores'] = scores
        return loss_dict

    @staticmethod
    def init_encoder_model(encoder_model_name: Union[str, Path], is_pickle: bool = False) -> PreTrainedModel:
        if is_pickle or str(encoder_model_name).endswith('.pkl') or str(encoder_model_name).endswith('.pt'):
            encoder_model = torch.load(encoder_model_name)
        else:
            logger = logging.getLogger('transformers.modeling_utils')
            logger.disabled = True
            encoder_model = AutoModelForPreTraining.from_pretrained(encoder_model_name).encoder
            logger.disabled = False
        return encoder_model

    @staticmethod
    def init_tokenizer(encoder_model_name: str):
        return AutoTokenizer.from_pretrained(encoder_model_name, use_fast=True)


class ConceptAdversarial(EncoderClassifier):
    def __init__(self, encoder_model: PreTrainedModel,
                 concepts_mapping: Union[str, Path, ConceptsMapping, Dict[str, List[str]]],
                 adversarial_concepts: List[str] = None,
                 concepts_to_predict: List[str] = None,
                 concept_fc_dict: Dict[str, Module] = None,
                 label_smoothing: float = 0.0,
                 ignore_unknown: bool = True,
                 adversarial_weight: float = 0.01):
        adversarial_concepts = adversarial_concepts.copy() if adversarial_concepts is not None else []
        if concepts_to_predict is not None:
            concepts_to_predict = concepts_to_predict.copy()
            concepts_to_predict += [ac for ac in adversarial_concepts if ac not in concepts_to_predict]
        else:
            concepts_to_predict = concepts_mapping.concepts.copy()
        super().__init__(encoder_model, concepts_mapping, concepts_to_predict, concept_fc_dict,
                         label_smoothing, ignore_unknown)
        self.adversarial_concepts = adversarial_concepts
        self.adversarial_weight = adversarial_weight

    def predict_token_scores(self, input_ids: LongTensor,
                             attention_mask: LongTensor,
                             with_encoder_grad: bool = False) -> Dict[str, Any]:
        hidden_states = self._extract_hidden_states(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    with_encoder_grad=with_encoder_grad)
        token_scores = {}
        for concept, concept_classifier in self.concept_fc_dict.items():
            if concept in self.adversarial_concepts:
                reversed_states = GradReverseLayerFunction.apply(hidden_states)
                token_scores[concept] = concept_classifier(reversed_states)
            else:
                token_scores[concept] = concept_classifier(hidden_states)
        return token_scores

    def predict_concepts_forward_step(self, input_ids: LongTensor,
                                      attention_mask: LongTensor,
                                      concepts_values: Dict[str, List[str]],
                                      ignore_unknown: bool = True,
                                      return_predictions: bool = False,
                                      return_probabilities: bool = False,
                                      return_scores: bool = False,
                                      with_encoder_grad: bool = True,
                                      **kwargs)-> Dict[str, Any]:
        loss_dict = super().predict_concepts_forward_step(input_ids, attention_mask, concepts_values, ignore_unknown,
                                                          return_predictions, return_probabilities, return_scores,
                                                          with_encoder_grad)
        losses = []
        weights = []
        for concept in self.concepts:
            loss = loss_dict[f'predict_{concept}_loss']
            if not torch.isnan(loss).any().item():
                losses.append(loss)
                weights.append(1.0 if concept not in self.adversarial_concepts else self.adversarial_weight)
        if len(weights) > 0:
            loss = sum([l * w for l, w in zip(losses, weights)]) / sum(weights)
        else:
            loss = torch.as_tensor(float('nan'))
        loss_dict.update({'loss': loss})
        return loss_dict


class LightningClassifier(LightningModule):
    def __init__(self,
                 # classifier args
                 output_dir: Union[str, Path],
                 concepts_mapping: Union[str, Path, ConceptsMapping],
                 encoder_model_name: str = T5_MODEL_NAME,

                 # model optimizer_args
                 optimizer_weight_decay: float = 1e-5,
                 optimizer_lr: float = 5e-5,
                 optimizer_eps: float = 1e-8,

                 # scheduler args (for optimizer)
                 training_steps: int = None,
                 warmup_steps: int = 0,

                 # more classifier args
                 concepts_to_predict: List[str] = None,
                 label_smoothing: float = 0.0,
                 ignore_unknown: bool = True,
                 adversarial_concepts: List[str] = None,
                 adversarial_weight: float = 0.01,
                 **kwargs):
        super().__init__()
        concepts_mapping_obj = ConceptsMapping.from_multiple_types(concepts_mapping)
        concepts_mapping = str(Path(output_dir) / 'concepts_mapping.json')
        concepts_mapping_obj.to_json(concepts_mapping)
        self.save_hyperparameters('output_dir', 'concepts_mapping', 'encoder_model_name',
                                  'optimizer_weight_decay', 'optimizer_lr', 'optimizer_eps',
                                  'training_steps', 'warmup_steps', 'concepts_to_predict', 'label_smoothing',
                                  'ignore_unknown', 'adversarial_concepts', 'adversarial_weight')
        encoder_model = EncoderClassifier.init_encoder_model(encoder_model_name)
        if adversarial_concepts is not None and len(adversarial_concepts) > 0:
            classifier = ConceptAdversarial(encoder_model=encoder_model,
                                            concepts_mapping=concepts_mapping_obj,
                                            concepts_to_predict=concepts_to_predict,
                                            concept_fc_dict=None,
                                            label_smoothing=label_smoothing,
                                            ignore_unknown=ignore_unknown,
                                            adversarial_concepts=adversarial_concepts,
                                            adversarial_weight=adversarial_weight)
        else:
            classifier = EncoderClassifier(encoder_model=encoder_model,
                                           concepts_mapping=concepts_mapping_obj,
                                           concepts_to_predict=concepts_to_predict,
                                           concept_fc_dict=None,
                                           label_smoothing=label_smoothing,
                                           ignore_unknown=ignore_unknown)
        self.classifier = classifier

    def get_modules(self) -> Dict[str, Any]:
        modules = self.classifier.get_modules()
        return modules

    def set_modules(self, encoder_model: PreTrainedModel = None,
                    concept_fc_dict: ModuleDict = None,
                    **kwargs):
        self.classifier.set_modules(encoder_model, concept_fc_dict)

    def _batch_forward_step(self, batch: Dict[str, Any],
                            return_predictions: bool = False,
                            return_probabilities: bool = False) -> Dict[str, Any]:
        step_kwargs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask',
                                                               'counterfactual_input_ids',
                                                               'counterfactual_attention_mask',
                                                               'concepts_values', 'ignore_unknown', 'with_encoder_grad']}
        return self.classifier.predict_concepts_forward_step(return_predictions=return_predictions,
                                                             return_probabilities=return_probabilities,
                                                             return_scores=False,
                                                             **step_kwargs)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        loss_dict = self._batch_forward_step(batch, return_predictions=False)
        self.log('train_loss', loss_dict['loss'], on_step=False, on_epoch=False, prog_bar=True, logger=True)
        return loss_dict

    def _epoch_end(self, outputs: List[Any], mode: str = 'train') -> Dict[str, Any]:
        epoch_loss_dict = {}
        concepts = self.classifier.concepts
        losses_names = ['predict_loss', 'predict_acc']
        losses_names += [f'predict_{k}_acc' for k in concepts] + [f'predict_{k}_loss' for k in concepts]
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
        if mode in ['test', 'validation']:
            to_save_keys = ['example_id'] + concepts + [f'{c}_prediction' for c in concepts]
            to_save_outputs = [subset_dict(batch, to_save_keys) for batch in outputs]
            to_save_outputs = concat_dicts(to_save_outputs)
            self.write_outputs(to_save_outputs, mode)
        return epoch_loss_dict

    def predictions_file_path(self, mode: str):
        output_dir = Path(self.hparams.output_dir)
        output_dir.mkdir(exist_ok=True)
        return output_dir / f"{mode}_predictions.csv"

    def predictions_df(self, mode: str):
        path = self.predictions_file_path(mode)
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Could not read outputs, encounter Exception at self.write_outputs(): "
                  f"{type(e)}:{e}")
            return None

    def write_outputs(self, outputs_dict: Dict[str, List[Any]], mode: str = 'test'):
        path = self.predictions_file_path(mode)
        try:
            df = pd.DataFrame(outputs_dict)
            for concept in self.classifier.concepts:
                df[f'{concept}_correct'] = df[f'{concept}'] == df[f'{concept}_prediction']
            df.to_csv(path, index=False)
        except Exception as e:
            print(f"Could not write outputs, encounter Exception at self.write_outputs(): "
                  f"{type(e)}:{e}")

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self._epoch_end(outputs, mode='train')

    def _eval_step(self, batch: Dict[str, Any]):
        batch_outputs = self._batch_forward_step(batch, return_predictions=True)
        predictions = batch_outputs.pop('predictions')
        batch_outputs['example_id'] = batch['example_id']
        for concept in batch['concepts_values']:
            batch_outputs[concept] = batch['concepts_values'][concept]
        for concept in predictions:
            batch_outputs[f'{concept}_prediction'] = predictions[concept]
        return batch_outputs

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._eval_step(batch)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self._epoch_end(outputs, mode='validation')

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._eval_step(batch)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self._epoch_end(outputs, mode='test')

    def configure_adamw_with_decay(self, weight_decay: float = None,
                                   lr: float = None,
                                   eps: float = None):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            # encoder model parameters without bias and LayerNorm
            {
                "params": [p for n, p in self.classifier.encoder_model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            # encoder model bias and LayerNorm
            {
                "params": [p for n, p in self.classifier.encoder_model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            # other parameters
            {
                "params": [p for n, p in self.classifier.concept_fc_dict.named_parameters()],
                "weight_decay": weight_decay, "lr": lr * 5
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
    def load_from_checkpoint_workaround(cls, checkpoint_path,
                                        map_location=None,
                                        **kwargs):
        checkpoint_obj = torch.load(checkpoint_path)
        hparams_dict = checkpoint_obj['hyper_parameters']
        hparams = Namespace(**hparams_dict)
        model = cls.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                         map_location=map_location,
                                         hparams=hparams, **kwargs)
        return model
