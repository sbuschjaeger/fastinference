#!/usr/bin/env python3

import itertools
import sys
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score

import json
import torch.onnx
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.autograd import Function

from test_utils import test_implementations

#from fastinference.Helper import NumpyEncoder

def sanatize_onnx(model):
    """ONNX does not support binary layers out of the box and exporting custom layers is sometimes difficult. This function sanatizes a given MLP so that it can be exported into an onnx file. To do so, it replaces all BinaryLinear layer with regular nn.Linear layers and BinaryTanh with Sign() layers. Weights and biases are copied and binarized as required.

    Args:
        model: The pytorch model.

    Returns:
        Model: The pytorch model in which each binary layer is replaced with the appropriate float layer.
    """

    # Usually I would use https://pytorch.org/docs/stable/generated/torch.heaviside.html for exporting here, but this is not yet supported in ONNX files. 
    class Sign(nn.Module):
        def forward(self, input):
            return torch.where(input > 0, torch.tensor([1.0]), torch.tensor([-1.0]))
            # return torch.sign(input)

    for name, m in reversed(model._modules.items()):
        print("Checking {}".format(name))

        if isinstance(m, BinaryLinear):
            print("Replacing {}".format(name))
            # layer_old = m
            layer_new = nn.Linear(m.in_features, m.out_features, hasattr(m, 'bias'))
            if (hasattr(m, 'bias')):
                layer_new.bias.data = binarize(m.bias.data)
            layer_new.weight.data = binarize(m.weight.data)
            model._modules[name] = layer_new

        if isinstance(m, BinaryTanh):
            model._modules[name] = Sign()

    return model

class BinarizeF(Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input > 0] = 1
        output[input <= 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #return grad_output, None
        grad_input = grad_output.clone()
        return grad_input#, None

# aliases
binarize = BinarizeF.apply

class BinaryLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BinaryLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            binary_weight = binarize(self.weight)

            return F.linear(input, binary_weight)
        else:
            binary_weight = binarize(self.weight)
            binary_bias = binarize(self.bias)
            return F.linear(input, binary_weight, binary_bias)

class BinaryTanh(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh(*args, **kwargs)

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarize(output)
        return output

class SimpleMLP(pl.LightningModule):

    def __init__(self, input_dim, n_classes, binarize = False, outpath = "."):
        super().__init__()
        # mnist images are (1, 28, 28) (channels, width, height) 
        if binarize:
            self.layer_1 = BinaryLinear(input_dim, 128)
            self.bn_1 = nn.BatchNorm1d(128)
            self.activation_1 = BinaryTanh()
            self.layer_2 = BinaryLinear(128, 256)
            self.bn_2 = nn.BatchNorm1d(256)
            self.activation_2 = BinaryTanh()
            self.layer_3 = BinaryLinear(256, n_classes)
        else:
            self.layer_1 = torch.nn.Linear(input_dim, 8)
            self.bn_1 = nn.BatchNorm1d(8)
            self.activation_1 = nn.ReLU()
            self.layer_2 = torch.nn.Linear(8, 16)
            self.bn_2 = nn.BatchNorm1d(16)
            self.activation_2 = nn.ReLU()
            self.layer_3 = torch.nn.Linear(16, n_classes)
        self.outpath = outpath
        self.input_dim = input_dim
        self.n_classes = n_classes

    def forward(self, x):
        # batch_size, _ = x.size()
        # x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = self.bn_1(x)
        x = self.activation_1(x)
        x = self.layer_2(x)
        x = self.bn_2(x)
        x = self.activation_2(x)
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict(self, X):
        tmp = TensorDataset(torch.Tensor(X)) 
        loader = DataLoader(tmp, batch_size=32) 

        all_preds = []
        for bdata in loader:
            x = bdata[0]
            preds =  self.forward(x.float()).argmax(axis=1)   
            all_preds.append(np.array(preds))

        return np.concatenate(all_preds) 

    def on_epoch_start(self):
        print('\n')

    def fit(self, X, y):
        XTrainT = torch.from_numpy(X).float()
        YTrainT = torch.from_numpy(y).long()

        train_dataloader = DataLoader(TensorDataset(XTrainT, YTrainT), batch_size=64)
        val_loader = None 

        trainer = pl.Trainer(max_epochs = 1, default_root_dir = self.outpath, progress_bar_refresh_rate = 0)
        trainer.fit(self, train_dataloader, val_loader)
        self.eval()

    def store(self, out_path, accuracy, model_name):
        dummy_x = torch.randn(1, self.input_dim, requires_grad=False)

        djson = {
            "accuracy":accuracy,
            "name":model_name
        }

        with open(os.path.join(out_path, model_name + ".json"), "w") as outfile:  
            json.dump(djson, outfile) #, cls=NumpyEncoder

        onnx_path = os.path.join(out_path,model_name+".onnx")
        print("Exporting {} to {}".format(model_name,onnx_path))
        model = sanatize_onnx(self)
        # https://github.com/pytorch/pytorch/issues/49229
        # set torch.onnx.TrainingMode.PRESERVE
        torch.onnx.export(model, dummy_x,onnx_path, training=torch.onnx.TrainingMode.PRESERVE, export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
        
        return onnx_path
        
def main():
    parser = argparse.ArgumentParser(description='Benchmark various mlp optimizations on the MNIST supplied dataset.')
    parser.add_argument('--outpath', required=True, help='Folder where data should written to.')
    parser.add_argument('--modelname', required=False, default="model", help='Modelname')
    parser.add_argument('--split','-s', required=False, default=0.2, type=float, help='Test/Train split.')
    parser.add_argument('--dataset','-d', required=True, help='Dataset to to be downloaded and used. Currently supported are {magic, mnist, fashion, eeg}.')
    parser.add_argument("--binarize", "-b", required=False, action='store_true', help="Trains a binarized neural network if true.")
    args = parser.parse_args()

    if args.dataset == "eeg":
        n_features, n_classes = 13,2
    elif args.dataset == "magic":
        n_features, n_classes = 10,2
    elif args.dataset in ["mnist", "fashion"]:
        n_features, n_classes = 28*28,10
    else:
        print("Only {eeg, magic, mnist, fashion} is supported for the --dataset/-d argument but you supplied {}.".format(args.dataset))
        sys.exit(1)

    model = SimpleMLP(n_features, n_classes, args.binarize, args.outpath)

    implementations = [ 
        ("NHWC",{"feature_type":"double"})
    ]
    
    if args.binarize:
        implementations.append( ("binary",{}) )


    optimizers = [
        ([None], [{}])
    ]

    performance = test_implementations(model = model, dataset= args.dataset, split = args.split, implementations = implementations, base_optimizers = optimizers, out_path = args.outpath, model_name = args.modelname)
    df = pd.DataFrame(performance)
    print(df)

if __name__ == '__main__':
    main()
