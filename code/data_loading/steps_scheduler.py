from typing import Dict, Union, Tuple, Any
import numpy as np


class StepsScheduler:
    def __init__(self, start_probabilities: Union[Tuple[Tuple[str, float]], Dict[str, float]] = None,
                 end_probabilities: Union[Tuple[Tuple[str, float]], Dict[str, float]] = None,
                 n_steps: int = None, seed: int = None):
        if isinstance(start_probabilities, (list, tuple)):
            start_probabilities = dict(start_probabilities)
        elif start_probabilities is None:
            start_probabilities = {}
        if isinstance(end_probabilities, (list, tuple)):
            end_probabilities = dict(end_probabilities)
        assert end_probabilities is not None and len(end_probabilities) > 0, "end_probabilities must contain at least" \
                                                                             " one key-probability pair"
        self.seed = seed
        self.rnp = np.random.RandomState(self.seed)
        self.set_training_probabilities_scheduling(start_probabilities, end_probabilities, n_steps)

    @staticmethod
    def normalize_probabilities(probabilities: Dict[str, float]) -> Dict[str, float]:
        values = list(probabilities.values())
        probabilities_sum = sum(values)
        assert sum([1 if v < 0 else 0 for v in values]) == 0, "All probabilities values must be non-negative."
        assert probabilities_sum > 0, "Probabilities sum should be positive"
        return {k: v / probabilities_sum for k, v in probabilities.items()}

    def set_training_probabilities(self, probabilities: Dict[str, float]):
        self.probabilities = StepsScheduler.normalize_probabilities(probabilities)

    def set_training_probabilities_scheduling(self, start_probabilities: Dict[str, float],
                                              end_probabilities: Dict[str, float],
                                              n_steps: int):
        self.current_step = 0
        self.n_steps = n_steps
        start_probabilities = start_probabilities.copy()
        end_probabilities = end_probabilities.copy()
        p_names = set(start_probabilities).union(set(end_probabilities))
        for p_name in p_names:
            # first we get the end_p. if it is None or not in end_probabilities, it will be set to 0.0
            end_p = end_probabilities.get(p_name, None)
            end_p = end_p if end_p is not None else 0.0
            # if p_name is not at start_probabilities or it is None, we set start_p to be end_p
            start_p = start_probabilities.get(p_name, None)
            start_p = start_p if start_p is not None else end_p
            start_probabilities[p_name] = start_p
            end_probabilities[p_name] = end_p
        start_probabilities = StepsScheduler.normalize_probabilities(start_probabilities)
        end_probabilities = StepsScheduler.normalize_probabilities(end_probabilities)
        if self.n_steps <= 0:
            start_probabilities = end_probabilities
            self.probabilities_growth = {p_name: 0.0 for p_name in p_names}
        else:
            self.probabilities_growth = {
                p_name: (end_probabilities[p_name] - start_probabilities[p_name]) / self.n_steps
                for p_name in p_names}
        self.set_training_probabilities(start_probabilities)

    def update_training_probabilities(self):
        if self.current_step < self.n_steps:
            probabilities = {p_name: p + self.probabilities_growth[p_name] for p_name, p in self.probabilities.items()}
            self.set_training_probabilities(probabilities)
            self.current_step += 1

    def sample_step_type(self, update: bool = True) -> str:
        step_types = []
        ps = []
        for step_type, p in self.probabilities.items():
            step_types.append(step_type)
            ps.append(p)
        if update:
            self.update_training_probabilities()
        return self.rnp.choice(step_types, 1, p=ps)[0]

    def retrieve_scheduler_kwargs(self) -> Dict[str, Any]:
        start_probabilities = self.probabilities
        n_steps = self.n_steps - self.current_step
        probabilities = {p_name: p + n_steps * self.probabilities_growth[p_name]
                         for p_name, p in self.probabilities.items()}
        end_probabilities = StepsScheduler.normalize_probabilities(probabilities)
        return {'start_probabilities': start_probabilities,
                'end_probabilities': end_probabilities,
                'n_steps': n_steps}
