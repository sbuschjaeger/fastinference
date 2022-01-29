import numpy as np
import pandas as pd

from PyPruning.Papers import create_pruner

def optimize(model, x_prune = None, y_prune = None, file = "", pruning_method = "", **kwargs):
    """Performs ensemble pruning on the given model given the data. Pruning is implemented via the PyPruning package (https://github.com/sbuschjaeger/PyPruning).
    For pruning either :code:`x_prune` / :code:`y_prune` as a pruning set must be provided or :code:`file` must point to a CSV file which has a "y" column. All remaining columns are interpreted as features. If both are provided then :code:`x_prune` / :code:`y_prune` is used before the file. If none are provided an error is thrown. 
    The specific pruning method can be chosen via :code:`pruning_method` and all additional arguments are passed to this function. For more details see https://sbuschjaeger.github.io/PyPruning/html/papers.html

    You can activate this optimization by simply passing :code:`"pruning"` to the optimizer, e.g.

    .. code-block::

        loaded_model = fastinference.Loader.model_from_file("/my/nice/tree.json")
        loaded_model.optimize("pruning", {"method":"reduced_error", "n_estimators":5})

    Args:
        model (Ensemble): The ensemble to be pruned
        x_prune (numpy.array, optional): A (N, d) matrix where N is the number of examples and d is the number of features. Defaults to None.
        y_prune (numpy.array, optional): A (N) vector which contains the labels (int values) for the corresponding features. Defaults to None.
        file (str, optional): Path to a CSV file from which x_prune/y_prune is loded if these are not provided. If set, the CSV must contain a "y" column to properly load y_prune. All remaining columns are interpreted as features. Defaults to "".
        pruning_method (str, optional): The pruning method used for pruning. For more details see https://sbuschjaeger.github.io/PyPruning/html/papers.html. Defaults to "".
    Returns:
        Ensemble: The pruned ensemble.
    """
    assert (x_prune is not None and y_prune is not None) or file.endswith(".csv")

    if x_prune is None:
        df = pd.read_csv(file)
        df = df.dropna()
        y_prune = df.pop("y")
        df = pd.get_dummies(df)
        x_prune = df.values

    pruner = create_pruner(pruning_method, **kwargs)
    pruner.prune(x_prune, y_prune, model.models, n_classes = model.n_classes)

    model.models = pruner.estimators_
    model.weights = pruner.weights_
    return model
