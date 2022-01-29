from pyexpat import model
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from tqdm import tqdm

from fastinference.models import Ensemble, Tree

def create_mini_batches(inputs, targets, batch_size, shuffle=False):
    """ Create an mini-batch like iterator for the given inputs / target / data. Shamelessly copied from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    
    Parameters
    ----------
    inputs : array-like vector or matrix 
        The inputs to be iterated in mini batches
    targets : array-like vector or matrix 
        The targets to be iterated in mini batches
    batch_size : int
        The mini batch size
    shuffle : bool, default False
        If True shuffle the batches 
    """
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    
    start_idx = 0
    while start_idx < len(indices):
        if start_idx + batch_size > len(indices) - 1:
            excerpt = indices[start_idx:]
        else:
            excerpt = indices[start_idx:start_idx + batch_size]
        
        start_idx += batch_size

        yield inputs[excerpt], targets[excerpt]

def refine(weights, estimators, X, Y, epochs, lr, batch_size, optimizer, verbose):
    """Performs SGD using the MSE loss over the leaf nodes of the given trees on the given data. The weights of each tree are respected during optimization but not optimized. 

    Args:
        weights (np.array): The weights of the trees.
        trees (list of Tree): The trees.
        X (2d np.array): The data.
        Y (np.array): The targe.
        epochs (int): The number of epochs SGD is performed.
        lr (float): The learning rate of SGD.
        batch_size (int): The batch size of SGD
        optimizer (str): The optimizer used for optimization. Can be {{"sgd", "adam"}}.
        verbose (bool): If True outputs the loss during optimization.

    Returns:
        list of trees: The refined trees.
    """
    n_classes = estimators[0].n_classes

    if batch_size > X.shape[0]:
        if verbose:
            print("WARNING: The batch size for SGD is larger than the dataset supplied: batch_size = {} > X.shape[0] = {}. Using batch_size = X.shape[0]".format(batch_size, X.shape[0]))
        batch_size = X.shape[0]
    
    if optimizer == "adam":
        m = np.zeros_like(weights)
        v = np.zeros_like(weights)
        t = 1

    for epoch in range(epochs):
        mini_batches = create_mini_batches(X, Y, batch_size, True) 

        batch_cnt = 0
        loss_sum = 0
        accuracy_sum = 0

        with tqdm(total=X.shape[0], ncols=150, disable = not verbose) as pbar:
            for x,y in mini_batches:
                # Prepare the target and apply all trees
                target_one_hot = np.array( [ [1.0 if yi == i else 0.0 for i in range(n_classes)] for yi in y] )
                proba = []
                for e, w in zip(estimators,weights):
                    proba.append(w * e.predict_proba(x))
                proba = np.array(proba)
                fbar = proba.sum(axis=0)

                deriv = 2 * (fbar - target_one_hot) * 1.0 / x.shape[0] * 1.0 / n_classes
                grad = np.mean(proba*deriv,axis=(1,2))

                if optimizer == "sgd":
                    # sgd
                    weights -= lr*grad 
                else:
                    # adam
                    beta1 = 0.9
                    beta2 = 0.999
                    m = beta1 * m + (1-beta1) * grad 
                    v = beta2 * v + (1-beta2) * (grad ** 2) 
                    m_corrected = m / (1-beta1**t)
                    v_corrected = v / (1-beta2**t)
                    weights -= lr * m_corrected / (np.sqrt(v_corrected) + 1e-8)
                    t += 1

                # compute some statistics 
                loss_sum += ((fbar - target_one_hot)**2).mean()
                accuracy_sum += (fbar.argmax(axis=1) == y).mean() * 100.0

                batch_cnt += 1 
                pbar.update(x.shape[0])
                
                desc = '[{}/{}] loss {:2.4f} accuracy {:2.4f}'.format(
                    epoch, 
                    epochs-1, 
                    loss_sum / batch_cnt,
                    accuracy_sum / batch_cnt, 
                )
                pbar.set_description(desc)

    return weights

def optimize(model, X = None, Y = None, file = "", epochs = 5, lr = 1e-2, batch_size = 32, optimizer = "adam", verbose = False, **kwargs):
    """Performs weight refinement of the given ensemble. Weight-refinement refines the weights of the estimators in the ensemble by optimizing the a joint loss function using SGD. The main purpose is to offer a refinement method that does not have any other dependencies and can be used "as is" with any kind of ensemble. This implementation only supports the MSE loss. If you are interested in other loss functions and/or other optimizers besides vanilla SGD and ADAM please have a look at TODO

    For refinement either :code:`X` / :code:`Y` must be provided or :code:`file` must point to a CSV file which has a "y" column. All remaining columns are interpreted as features. If both are provided then :code:`X` / :code:`Y` is used before the file. If none are provided an error is thrown. 

    You can activate this optimization by simply passing :code:`"leaf-refinement"` to the optimizer, e.g.

    .. code-block::

        loaded_model = fastinference.Loader.model_from_file("/my/nice/tree.json")
        loaded_model.optimize("leaf-refinement", {"X": some_data, "Y" : some_targets})

    Args:
        model (Ensemble of Trees or Tree): The Tree or Ensemble of Trees that should be refined
        X (2d np.array, optional): A (N,d) data matrix used for refinement. Defaults to None.
        Y (np.array, optional): A (N,) target vector used for refinement. Defaults to None.
        file (str, optional): Path to a CSV file from which X/Y is loaded if these are not provided. If set, the CSV must contain a "y" column to properly load Y. All remaining columns are interpreted as features. Defaults to "".
        epochs (int, optional): Number of epochs used for SGD/ADAM. Defaults to 5.
        lr (float, optional): Learning rate used for SGD/ADAM. Defaults to 1e-1.
        batch_size (int, optional): Batch size used for SGD/ADAM. Defaults to 32.
        optimizer (str, optional): Optimizer for optimization. Can be {{"sgd", "adam"}}. Defaults to "adam".
        verbose (bool, optional): If True outputs the loss during optimization. Defaults to False.

    Returns:
        Ensemble of Trees or Tree: The refined ensemble / tree
    """    
    assert (X is not None and Y is not None) or file.endswith(".csv"), "You can either supply (X,y) directly or use `file' to supply a csv file that contains the data. You did not provide either. Please do so."

    assert isinstance(model, Ensemble.Ensemble), "Weight refinement only works for Ensembles, but you provided {}".format(model.__class__.__name__)

    assert lr >= 0, "Learning rate must be positive, but you gave {}".format(lr)
    assert epochs >= 1, "Number of epochs must be >= 1, but you gave {}".format(epochs)
    assert optimizer in ["sgd", "adam"], "The optimizer must be from {{adam, sgd}}, but you gave {}".format(optimizer)
    
    if X is None or Y is None:
        df = pd.read_csv(file)
        df = df.dropna()
        Y = df.pop("y")
        df = pd.get_dummies(df)
        X = df.values

    if batch_size > X.shape[0]:
        print("Warning: batch_size is greater than supplied number of datapoints. {} > {}. Setting batch_size = {}".format(batch_size,X.shape[0], X.shape[0]))
        batch_size = X.shape[0]

    model.weights = refine(model.weights, model.models, X, Y, epochs, lr, batch_size, optimizer, verbose)
    return model