from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from tqdm import tqdm

from fastinference.models import Ensemble, Tree
import torch
import torch.nn as nn

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

class WeightRefinery(nn.Module):

    def __init__(self, epochs, lr, batch_size, optimizer, verbose,loss_function = "mse", loss_type = "upper", l_reg = 1.0):
        super().__init__()

        assert loss_function in ["mse", "nll", "cross-entropy"], "LeafRefinery only supports the {{mse, nll, cross-entropy}} loss but you gave {}".format(loss_function)
        assert lr >= 0, "Learning rate must be positive, but you gave {}".format(lr)
        assert epochs >= 1, "Number of epochs must be >= 1, but you gave {}".format(epochs)
        assert optimizer in ["sgd", "adam"], "The optimizer must be from {{adam, sgd}}, but you gave {}".format(optimizer)
        
        if loss_type == "exact":
            assert 0 <= l_reg <= 1, "You set loss_type to exact. In this case l_reg should be from [0,1], but you supplied l_reg = {}".format(l_reg)

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.verbose = verbose
        self.loss_function = loss_function
        self.loss_type = loss_type
        self.l_reg = l_reg

    def _loss(self, pred, target):
        if self.loss_function == "mse":
            target_one_hot = torch.nn.functional.one_hot(target, num_classes = pred.shape[1]).double()
            return torch.nn.functional.mse_loss(pred, target_one_hot)
        elif self.loss_function == "nll":
            return torch.nn.functional.nll_loss(pred, target)
        elif self.loss_function == "cross-entropy":
            return torch.nn.functional.cross_entropy(pred, target)
        else:
            raise ValueError("Unknown loss function set in LeafRefinery")

    def compute_loss(self, fbar, base_preds, target):
        if self.loss_type == "upper":
            n_classes = fbar.shape[1]
            n_preds = fbar.shape[0]
            D = torch.eye(n_classes).repeat(n_preds, 1, 1).double()
        else:
            if self.loss_function == "mse":
                n_classes = fbar.shape[1]
                n_preds = fbar.shape[0]

                eye_matrix = torch.eye(n_classes).repeat(n_preds, 1, 1).double()
                D = 2.0*eye_matrix
            elif self.loss_function == "nll":
                n_classes = fbar.shape[1]
                n_preds = fbar.shape[0]
                D = torch.eye(n_classes).repeat(n_preds, 1, 1).double()
                target_one_hot = torch.nn.functional.one_hot(target, num_classes = n_classes)

                eps = 1e-7
                diag_vector = target_one_hot*(1.0/(fbar**2+eps))
                D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
            elif self.loss_function == "cross-entropy":
                n_preds = fbar.shape[0]
                n_classes = fbar.shape[1]
                f_bar_softmax = nn.functional.softmax(fbar,dim=1)

                D = -1.0*torch.bmm(f_bar_softmax.unsqueeze(2), f_bar_softmax.unsqueeze(1)).double()
                diag_vector = f_bar_softmax*(1.0-f_bar_softmax)
                D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
            else:
                # NOTE: We should never reach this code path
                raise ValueError("Invalid combination of mode and loss function in Leaf-refinement.")

        f_loss = self._loss(fbar, target)
        losses = []
        n_estimators = len(base_preds)
        for pred in base_preds:
            diff = pred - fbar
            covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
            div = 1.0/n_estimators * 1.0/2.0 * covar

            i_loss = self._loss(pred, target)

            if self.loss_type == "exact":
                # Eq. (4)
                reg_loss = 1.0/n_estimators * i_loss - self.l_reg * div
            else:
                # Eq. (5) where we scale the ensemble loss with 1.0/self.n_estimators due to the summation later
                reg_loss = 1.0/n_estimators * self.l_reg * f_loss + (1.0 - self.l_reg)/n_estimators * i_loss
            
            losses.append(reg_loss)
        return torch.stack(losses).sum()

    def refine(self, weights, estimators, X, Y):
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
        if self.batch_size > X.shape[0]:
            if self.verbose:
                print("WARNING: The batch size for SGD is larger than the dataset supplied: batch_size = {} > X.shape[0] = {}. Using batch_size = X.shape[0]".format(self.batch_size, X.shape[0]))
            self.batch_size = X.shape[0]

        n_estimators = len(estimators)
        self.weights = nn.Parameter(torch.tensor(weights))

        # Train the model
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr) 

        for epoch in range(self.epochs):
            mini_batches = create_mini_batches(X, Y, self.batch_size, True) 

            batch_cnt = 0
            loss_sum = 0
            accuracy_sum = 0

            with tqdm(total=X.shape[0], ncols=150, disable = not self.verbose) as pbar:
                for x,y in mini_batches:
                    
                    # Prepare the target and apply all trees
                    proba = []
                    for e, w in zip(estimators,self.weights):
                        p = torch.tensor(e.predict_proba(x))
                        proba.append(w * p)
                    proba = torch.stack(proba)
                    fbar = proba.mean(axis=0)

                    # Do the actual optimization
                    loss = self.compute_loss(fbar, proba, torch.tensor(y))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # compute some statistics 
                    loss_sum += loss.detach().numpy()
                    accuracy_sum += accuracy_score(fbar.argmax(axis=1),y) * 100.0

                    batch_cnt += 1 
                    pbar.update(x.shape[0])
                    
                    desc = '[{}/{}] loss {:2.4f} accuracy {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        loss_sum / batch_cnt,
                        accuracy_sum / batch_cnt, 
                    )
                    pbar.set_description(desc)

        return 1.0 / n_estimators * self.weights.detach().numpy()

def optimize(model, X = None, Y = None, file = "", epochs = 5, lr = 1e-1, batch_size = 32, optimizer = "adam", verbose = False, loss_function = "mse", loss_type = "upper", l_reg = 1.0, **kwargs):
    """Performs leaf refinement in the given tree ensemble with PyTorch. Leaf-refinement refines the probability estimates in the leaf nodes of each tree by optimizing the NCL loss function using SGD. This implementation uses PyTorch to compute the gradient of the loss function and is thereby a little more flexible. Its main purpose is to facilitate easy development of new refinement methods.  

    For refinement either :code:`X` / :code:`Y` must be provided or :code:`file` must point to a CSV file which has a "y" column. All remaining columns are interpreted as features. If both are provided then :code:`X` / :code:`Y` is used before the file. If none are provided an error is thrown. 

    You can activate this optimization by simply passing :code:`"leaf-refinement-pytorch"` to the optimizer, e.g.

    .. code-block::

        loaded_model = fastinference.Loader.model_from_file("/my/nice/tree.json")
        loaded_model.optimize("leaf-refinement", {"X": some_data, "Y" : some_targets})

    Reference:
        Buschjäger, Sebastian, and Morik, Katharina "There is no Double-Descent in Random Forests" 2021 (https://arxiv.org/abs/2111.04409)

        Buschjäger, Sebastian, Pfahler, Lukas, and Morik, Katharina "Generalized Negative Correlation Learning for Deep Ensembling" 2020 (https://arxiv.org/pdf/2011.02952.pdf)

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
        loss_function (str, optional): The loss to be optimized. Can be {{"mse","nll","cross-entropy"}}. Defaults to "mse"
        loss_type (str, optional): The way the loss is interpreted which influences the interpretation of the l_reg parameter. Can be {{"upper", "exact"}}. Defaults to "upper". 
        l_reg (float, optional): The regularizer. If loss_type = "upper" is set then l_reg should be [0,1], where 0 indicates independent refinement of the trees and 1 the joint optimization of all trees. A value in-between is a mix of both approaches. If loss_type = "exact" you can freely choose l_reg, where l_reg < 0 actively discourages diversity across the trees, l_reg = 0 ignores it and l_reg > 1 promotes it. Defaults to 1.0

    Returns:
        Ensemble of Trees or Tree: The refined ensemble / tree
    """    
    assert (X is not None and Y is not None) or file.endswith(".csv"), "You can either supply (X,y) directly or use `file' to supply a csv file that contains the data. You did not provide either. Please do so."

    assert isinstance(model, (Tree.Tree, Ensemble.Ensemble)), "Leaf refinement does only work with Tree Ensembles or single trees, but you provided {}".format(model.__class__.__name__)

    assert lr >= 0, "Learning rate must be positive, but you gave {}".format(lr)
    assert epochs >= 1, "Number of epochs must be >= 1, but you gave {}".format(epochs)
    assert optimizer in ["sgd", "adam"], "The optimizer must be from {{adam, sgd}}, but you gave {}".format(optimizer)
    assert loss_function in ["mse", "nll", "cross-entropy"], "Leaf-Refinement-Pytorch only supports the {{mse, nll, cross-entropy}} loss but you gave {}".format(loss_function)
    
    if loss_type == "exact":
        assert 0 <= l_reg <= 1, "You set loss_type to exact. In this case l_reg should be from [0,1], but you supplied l_reg = {}".format(l_reg)

    if X is None or Y is None:
        df = pd.read_csv(file)
        df = df.dropna()
        Y = df.pop("y")
        df = pd.get_dummies(df)
        X = df.values

    if batch_size > X.shape[0]:
        print("Warning: batch_size is greater than supplied number of datapoints. {} > {}. Setting batch_size = {}".format(batch_size,X.shape[0], X.shape[0]))
        batch_size = X.shape[0]

    weight_refinery = WeightRefinery(epochs, lr, batch_size, optimizer, verbose, loss_function, loss_type, l_reg)
    model.weights = weight_refinery.refine(model.weights, model.models, X, Y)
    
    return model
