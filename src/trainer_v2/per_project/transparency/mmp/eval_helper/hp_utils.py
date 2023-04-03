import random
from typing import Dict, List
from typing import List, Iterable, Callable, Dict, Tuple, Set

from itertools import product



def generate_hyperparameters(param_range: Dict[str, Iterable[float]]) -> List[Dict[str, float]]:
    """
    Generates a list of dictionaries containing all possible combinations of hyperparameters.
    """
    hyperparameters_list = []

    # Generate all possible combinations of hyperparameters
    for param_values in product(*param_range.values()):
        hyperparameters = {}
        for param_name, param_value in zip(param_range.keys(), param_values):
            hyperparameters[param_name] = param_value
        hyperparameters_list.append(hyperparameters)

    return hyperparameters_list
