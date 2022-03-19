from pathlib import Path
from typing import List, Union, Dict

from project_utils.constants import UNK
from project_utils.functions import save_json, load_json


class ConceptsMapping(dict):
    def __init__(self, concepts_values: Dict[str, List[str]] = None):
        """
        An inherited class of dict, used to store the concepts as keys and a mapping of the concepts' possible values
        to indices: {'concept': {'from_index': [UNK, value1, value2, ...],
                                 'to_index': {0: UNK, 1: value1, 2: value2,...}}}
        Note that the first value of each concept is the unknown (UNK) value.
        :param concepts_values: a dict where the keys are the concepts and the values are lists containing their
        possible values.
        """
        super(ConceptsMapping, self).__init__()
        if concepts_values is not None:
            for concept, concept_values in concepts_values.items():
                for value in concept_values:
                    self.add_concept_value(concept, value)

    @classmethod
    def from_multiple_types(cls, concepts_mapping: Union[str, Path, "ConceptsMapping", Dict[str, List[str]]]):
        if isinstance(concepts_mapping, (str, Path)):
            concepts_mapping = ConceptsMapping.from_json(concepts_mapping)
        elif isinstance(concepts_mapping, ConceptsMapping):
            concepts_mapping = concepts_mapping
        elif isinstance(concepts_mapping, dict):  # concepts_mapping = Dict[str, List[str]
            concepts_mapping = cls(concepts_mapping)
        else:
            raise ValueError(f"`concepts_mapping` should be a json path, a ConceptsMapping object"
                             f" or a dict of concepts-values")
        return concepts_mapping

    @classmethod
    def from_json(cls, json_path: Union[str, Path]):
        kwargs = load_json(json_path)
        cm = cls()
        cm.update(kwargs)
        return cm

    def to_json(self, json_path: Union[str, Path]):
        save_json({c: v for c, v in self.items()}, json_path)

    def __setitem__(self, key, value):
        self.add_concept_value(key, value)

    @property
    def concepts(self) -> List[str]:
        return list(self.keys())

    @property
    def concepts_values(self) -> Dict[str, List[str]]:
        return {c: self[c]['from_index'] for c in self.concepts}

    def value_to_index(self, concept: str, value: str, ignore_unknown: bool = False) -> int:
        value = value if value is not None else UNK
        index = self[concept]['to_index'].get(value, 0)
        return index if not ignore_unknown else index - 1

    def index_to_value(self, concept: str, index: int, ignore_unknown: bool = False) -> str:
        index = index if not ignore_unknown else index + 1
        return self[concept]['from_index'][index]

    def concept_values(self, concept: str) -> List[str]:
        return self[concept]['from_index']

    def add_concept_value(self, concept: str = None, value: str = None):
        if concept is not None and concept not in self:
            if value is None or value == UNK:
                concept_values = [UNK]
            else:
                concept_values = [UNK, value]
            mapping = {'to_index': {v: i for i, v in enumerate(concept_values)},
                       'from_index': concept_values}
            super(ConceptsMapping, self).__setitem__(concept, mapping)
        elif value is not None and value not in self[concept]['from_index']:
            self[concept]['to_index'][value] = len(self.concept_values(concept))
            self[concept]['from_index'] += [value]
        else:
            pass
