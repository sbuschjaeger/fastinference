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

def apply(tree, mapping, X):
    """Applies the given tree to the given data by using the provided leaf-node mapping. To do so traverse the tree as usual and extracts the relevant leaf node indices

    Args:
        tree (Tree): The tree that should be applied 
        mapping (dict): The mapping of leaf node indices (e.g. node.id) and their counterpart in the leaf array used by SGD
        X (2d np.array): The data matrix

    Returns:
        np.array: (N,) numpy array with the leaf index for each data item 
    """
    if len(X.shape) == 1:
        X = X.reshape(1,X.shape[0])
    
    ids = []
    for x in X:
        node = tree.head

        while(node.prediction is None):
            if (x[node.feature] <= node.split): 
                node = node.leftChild
            else:
                node = node.rightChild
        ids.append(mapping[node.id])

    return np.array(ids)

def refine(weights, trees, X, Y, epochs, lr, batch_size, optimizer, verbose):
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
    n_classes = trees[0].n_classes

    if batch_size > X.shape[0]:
        if verbose:
            print("WARNING: The batch size for SGD is larger than the dataset supplied: batch_size = {} > X.shape[0] = {}. Using batch_size = X.shape[0]".format(batch_size, X.shape[0]))
        batch_size = X.shape[0]

    # To make the following SGD somewhat efficient this code extracts all the leaf nodes and gathers them in an array. To do so it iterates over all trees and all nodes in the trees. Each leaf node is added to the leafs array and the corresponding node.id is stored in mappings. For scikit-learn trees this would be much simpler as they already offer a dedicated leaf field:
    # leafs = []
    # for tree in trees:
    #     tmp = tree.tree_.value / tree.tree_.value.sum(axis=(1,2))[:,np.newaxis,np.newaxis]
    #     leafs.append(tmp.squeeze(1))
    mappings = []
    leafs = []
    for t, w in zip(trees, weights):
        leaf_mapping = {}
        l = []
        for i, n in enumerate(t.nodes):
            if n.prediction is not None:
                leaf_mapping[n.id] = len(l)
                # Normalize the values in the leaf nodes for SGD. This is usually a better initialization
                pred = np.array(n.prediction) / sum(n.prediction)
                l.append(pred)
        mappings.append(leaf_mapping)
        leafs.append(np.array(l))
    
    if optimizer == "adam":
        m = []
        v = []
        t = 1
        for l in leafs:
            m.append(np.zeros_like(l))
            v.append(np.zeros_like(l))

    for epoch in range(epochs):
        mini_batches = create_mini_batches(X, Y, batch_size, True) 

        batch_cnt = 0
        loss_sum = 0
        accuracy_sum = 0

        with tqdm(total=X.shape[0], ncols=150, disable = not verbose) as pbar:
            for x,y in mini_batches:
                
                # Prepare the target and apply all trees
                target_one_hot = np.array( [ [1.0 if yi == i else 0.0 for i in range(n_classes)] for yi in y] )
                indices = [apply(t, m, x) for t,m in zip(trees, mappings)]
                pred = []
                for i, idx, w in zip(range(len(trees)), indices, weights):
                    pred.append(w * leafs[i][idx])
                pred = np.array(pred)
                fbar = pred.sum(axis=0)

                # SGD
                if optimizer == "sgd":
                    deriv = 2 * (fbar - target_one_hot) * 1.0 / x.shape[0] * 1.0 / n_classes #* 1.0 / len(trees) 
                    for i, idx in zip(range(len(trees)), indices):
                        np.add.at(leafs[i], idx, - lr * deriv)
                else:    
                    # Adam
                    deriv = 2 * (fbar - target_one_hot) * 1.0 / x.shape[0] * 1.0 / n_classes #* 1.0 / len(trees)
                    beta1 = 0.9
                    beta2 = 0.999
                    for i, idx in zip(range(len(trees)), indices):
                        grad = np.zeros_like(leafs[i])
                        np.add.at(grad, idx, deriv)
                        m[i] = beta1 * m[i] + (1-beta1) * grad 
                        v[i] = beta2 * v[i] + (1-beta2) * (grad ** 2)
                        m_corrected = m[i] / (1-beta1**t)
                        v_corrected = v[i] / (1-beta2**t)
                        leafs[i] += - lr * m_corrected / (np.sqrt(v_corrected) + 1e-8)
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

    # Copy the optimized leafs back into the trees with the pre-computed mapping 
    for t, m, l in zip(trees, mappings, leafs):
        for nid, i in m.items():
            t.nodes[nid].prediction = l[i].tolist()
    return trees

def optimize(model, X = None, Y = None, file = "", epochs = 5, lr = 1e-1, batch_size = 32, optimizer = "adam", verbose = False, **kwargs):
    """Performs leaf refinement in the given tree ensemble. Leaf-refinement refines the probability estimates in the leaf nodes of each tree by optimizing the a joint loss function using SGD. The main purpose is to offer a refinement method that does not have any other dependencies and can be used "as is". This implementation only supports the MSE loss. If you are interested in other loss functions and/or other optimizers besides vanilla SGD and ADAM please have a look at TODO

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

    assert isinstance(model, (Tree.Tree, Ensemble.Ensemble)), "Leaf refinement does only work with Tree Ensembles or single trees, but you provided {}".format(model.__class__.__name__)

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

    if isinstance(model, Tree.Tree):
        model = refine([1.0], [model], X, Y, epochs, lr, batch_size, optimizer, verbose)[0]
    else:
        model.models = refine(model.weights, model.models, X, Y, epochs, lr, batch_size, optimizer, verbose)

    return model
