import numpy as np

from fastinference.models import Linear

def optimize(model, **kwargs):
    """Merges all linear models in this ensemble into a single linear model. 

    Args:
        model (Ensemble): The ensemble

    Returns:
        Ensemble: The merged ensemble.
    """
    linear = [(m,w) for m,w in zip(model.models, model.weights) if isinstance(m, Linear.Linear)]
    non_linear = [(m,w) for m,w in zip(model.models, model.weights) if not isinstance(m, Linear.Linear)]

    if len(linear) > 0:
        new_weight = 0
        new_coef = np.zeros_like(linear[0][0].coef)
        new_intercept = np.zeros_like(linear[0][0].intercept)
        for m, w in linear:
            new_weight += w
            new_coef += m.coef

            new_intercept += m.intercept

        linear_model = linear[0][0]
        linear_model.coef = new_coef
        linear_model.intercept = new_intercept

        model.models = [m for m, _ in non_linear] + [linear_model]
        model.weights = [w for _, w in non_linear] + [new_weight]

    return model