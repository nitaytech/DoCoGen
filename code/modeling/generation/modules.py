from typing import Union, List, Dict

import torch
from torch import FloatTensor, LongTensor, Tensor
from torch.autograd import Function
from torch.nn import Module, Embedding, ModuleDict
from transformers import PreTrainedTokenizer
from transformers.generation_logits_process import LogitsProcessor
from project_utils.constants import DIM, UNK


class OrientationEmbedding(Embedding):
    def __init__(self, orientations: List[str], lm_embedding_module: Embedding, lm_tokenizer: PreTrainedTokenizer,
                 embedding_dim: int = DIM,  unknown_orientation: str = UNK, **embedding_kwargs):
        super().__init__(num_embeddings=len(orientations), embedding_dim=embedding_dim, **embedding_kwargs)
        self.unknown_orientation = unknown_orientation
        # if unknown_orientation is None, we don't use it.
        if self.unknown_orientation is not None:
            orientations = [self.unknown_orientation] + [o for o in orientations if o != self.unknown_orientation]
        self.orientation2idx = {o: i for i, o in enumerate(orientations)}
        # init the embeddings with the lm embeddings vector of the orientation:
        with torch.no_grad():
            for o, idx in self.orientation2idx.items():
                if idx == 0 and self.unknown_orientation is not None:
                    input_ids = torch.tensor([lm_tokenizer.pad_token_id],
                                             device=self.weight.device)
                else:
                    input_ids = torch.tensor(lm_tokenizer(o)['input_ids'][:-1],
                                             device=self.weight.device)
                weight = lm_embedding_module(input_ids).mean(dim=0).tolist()
                weight = torch.tensor(weight, device=self.weight.device, dtype=self.weight.dtype)
                self.weight[idx] = weight

    def forward(self, input: Tensor) -> Tensor:
        # check if the input is a list of string, i.e., orientations
        if isinstance(input, List) and isinstance(input[0], str):
            return self.orientations_forward(input)
        return super().forward(input)

    def orientations_forward(self, orientations: List[str]) -> Tensor:
        input = torch.tensor([self.orientation2idx[o] for o in orientations], device=self.weight.device)
        return self.forward(input)


class OrientationEmbeddingDict(ModuleDict):
    def __init__(self, concepts_dict: Dict[str, OrientationEmbedding]):
        super().__init__(concepts_dict)
        self.concept2idx = {c: i for i, c in enumerate(self.concepts)}

    @classmethod
    def from_concepts_orientations(cls, concepts_orientations: Dict[str, List[str]], lm_embedding_module: Embedding,
                                   lm_tokenizer: PreTrainedTokenizer, embedding_dim: int = DIM,
                                   unknown_orientation: str = UNK, **embedding_kwargs):
        concepts_dict = {}
        for concept, orientations in concepts_orientations.items():
            concepts_dict[concept] = OrientationEmbedding(orientations, lm_embedding_module, lm_tokenizer,
                                                          embedding_dim, unknown_orientation, **embedding_kwargs)
        return cls(concepts_dict)

    @property
    def concepts(self):
        return list(self.keys())

    def concept_order(self, concept: str):
        return self.concept2idx[concept]

    def get_orientations_embeddings(self, batch_size: int,
                                    orientations: Dict[str, List[str]],
                                    concepts_order: List[str] = None) -> Tensor:
        embeddings = []
        if concepts_order is None:
            concepts_order = self.concepts
        else:
            for concept in concepts_order:
                assert concept in self, f"Concept {concept} has no OrientationEmbedding"
        for concept in concepts_order:
            orientation_embedding = self[concept]
            if concept not in orientations:
                assert orientation_embedding.unknown_orientation is not None, \
                    f"Concept {concept} is used but is not in `orientations` and has no `unknown_orientation`"
                concept_orientations = [orientation_embedding.unknown_orientation for _ in range(batch_size)]
            else:
                concept_orientations = orientations[concept]
            embeddings.append(orientation_embedding(concept_orientations))
        embeddings = torch.stack(embeddings, dim=1)
        return embeddings


class OrientedProcessor(LogitsProcessor):
    def __init__(self, possible_ids: List[List[int]], filter_value: float = -float("inf")):
        max_len = max([len(ids) for ids in possible_ids])
        # padding the possible ids with zeros (should be the pad_token_id), so it will be possible to create a tensor
        possible_ids = [ids + ([0] * (max_len - len(ids))) for ids in possible_ids]
        self.possible_ids = torch.tensor(possible_ids)
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids are the generated_prompt, not the encoder input_ids
        self.possible_ids = self.possible_ids.to(device=input_ids.device)
        # check if beam search is used, if so, each input need to be repeated
        if input_ids.shape[0] != self.possible_ids.shape[0]:
            n, m = self.possible_ids.shape[0], input_ids.shape[0]
            assert m // n == m / n
            reps = torch.tensor([m // n] * n, device=self.possible_ids.device)
            self.possible_ids = torch.repeat_interleave(self.possible_ids, reps, dim=0)

        # init the mask with all True, then fill with False all the entries of possible_ids
        not_possible_mask = (scores == False) | (scores != False)
        not_possible_mask.scatter_(1, self.possible_ids, False)
        scores.masked_fill_(not_possible_mask, self.filter_value)
        return scores


class OrientationPossibleIds:
    def __init__(self, possible_words_per_orientation: Dict[str, Union[List[str], None]],
                 lm_tokenizer: PreTrainedTokenizer, unknown_orientation: str = UNK):
        self.all_ids = set(range(lm_tokenizer.vocab_size))
        self.special_ids = set([sid for sid in lm_tokenizer.all_special_ids if sid != lm_tokenizer.unk_token_id])
        possible_ids = {None: self.all_ids}
        if unknown_orientation is not None:
            self.unknown_orientation = unknown_orientation
            possible_ids[unknown_orientation] = self.all_ids
        else:
            self.unknown_orientation = None
        for orientation, possible_words in possible_words_per_orientation.items():
            if possible_words is None:
                possible_ids[orientation] = self.all_ids
            else:
                possible_words = ' '.join(possible_words)
                ids = set(lm_tokenizer(possible_words)['input_ids'][:-1])
                possible_ids[orientation] = ids.union(self.special_ids)
        self.possible_ids_per_orientation = possible_ids

    def prepare_possible_ids_for_generation(self, input_ids: LongTensor = None,
                                            orientations: List[str] = None) -> Union[None, List[List[int]]]:
        if orientations is None:
            return None
        possible_ids = [self.possible_ids_per_orientation[o] for o in orientations]
        if input_ids is not None:
            assert len(input_ids) == len(orientations), "`input_ids` should have the same size as `orientations`"
            for i, input_ids_set in enumerate([set(ids) for ids in input_ids.tolist()]):
                possible_ids[i] = possible_ids[i].union(input_ids_set)
        return [list(ids) for ids in possible_ids]

    def prepare_oriented_processor(self, input_ids: LongTensor = None,
                                   orientations: List[str] = None,
                                   filter_value: float = -float("inf")) -> Union[None, OrientedProcessor]:
        possible_ids = self.prepare_possible_ids_for_generation(input_ids, orientations)
        if possible_ids is None:
            return None
        else:
            return OrientedProcessor(possible_ids, filter_value)


class OrientationPossibleIdsDict:
    def __init__(self, possible_words_per_concept: Dict[str, Dict[str, Union[List[str], None]]],
                 lm_tokenizer: PreTrainedTokenizer, unknown_orientation: str = UNK):
        self.concept_possible_ids = {}
        for concept, possible_words_per_orientation in possible_words_per_concept.items():
            self.concept_possible_ids[concept] = OrientationPossibleIds(possible_words_per_orientation,
                                                                        lm_tokenizer, unknown_orientation)

    @property
    def concepts(self):
        return list(self.concept_possible_ids.keys())

    @property
    def unknown_orientation(self):
        return self.concept_possible_ids[self.concepts[0]].unknown_orientation

    @property
    def all_ids(self):
        return self.concept_possible_ids[self.concepts[0]].all_ids

    def prepare_possible_ids_for_generation(self, input_ids: LongTensor = None,
                                            orientations: Dict[str, List[str]] = None) -> Union[None, List[List[int]]]:
        if orientations is None:
            return None
        possible_ids = [set(ids) for ids in input_ids.tolist()]
        # If all the orientations of an example are unknowns, use all ids. Otherwise, use union of possible_ids
        all_unknowns = [True for _ in possible_ids]
        unknown_orientation = self.unknown_orientation
        all_ids = self.all_ids
        for concept, concept_orientations in orientations.items():
            for i, orientation in enumerate(concept_orientations):
                if orientation is not None and orientation != unknown_orientation:
                    all_unknowns[i] = False
                opi = self.concept_possible_ids[concept].possible_ids_per_orientation[orientation]
                possible_ids[i] = possible_ids[i].union(opi)
        for i, use_all in enumerate(all_unknowns):
            if use_all:
                possible_ids[i] = all_ids
        return [list(ids) for ids in possible_ids]

    def prepare_oriented_processor(self, input_ids: LongTensor = None,
                                   orientations: Dict[str, List[str]] = None,
                                   filter_value: float = -float("inf")) -> Union[None, OrientedProcessor]:
        possible_ids = self.prepare_possible_ids_for_generation(input_ids, orientations)
        if possible_ids is None:
            return None
        else:
            return OrientedProcessor(possible_ids, filter_value)


class LabelSmoothingLoss(Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    """

    taken from https://github.com/pytorch/pytorch/issues/7455

    """

    def forward(self, scores: FloatTensor, labels: LongTensor, n_classes: int, dim: int = -1):
        confidence = 1.0 - self.smoothing
        scores = torch.log_softmax(scores, dim=dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(scores)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, labels.data.unsqueeze(1), confidence)
        return torch.mean(torch.sum(-true_dist * scores, dim=dim))


class GradReverseLayerFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.alpha = 1.5
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None