#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import os
import argparse

import json
import torch.onnx
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
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

    for name, m in model._modules.items():
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

        if isinstance(m, BinaryConv2d):
            print("Replacing {}".format(name))
            # layer_old = m
            layer_new = nn.Conv2d(
                in_channels = m.in_channels, 
                out_channels = m.out_channels, 
                kernel_size = m.kernel_size, 
                stride = m.stride, 
                #padding = m.padding,
                bias = hasattr(m, 'bias')
            )

            if (hasattr(m, 'bias')):
                layer_new.bias.data = binarize(m.bias.data)
            layer_new.weight.data = binarize(m.weight.data)
            model._modules[name] = layer_new
        
        # if isinstance(m, nn.BatchNorm2d):
        #     layer_new = WrappedBatchNorm(m)
        #     model._modules[name] = layer_new

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

class BinaryConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(BinaryConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            binary_weight = binarize(self.weight)

            return F.linear(input, binary_weight)
        else:
            binary_weight = binarize(self.weight)
            binary_bias = binarize(self.bias)
            return F.conv2d(input, binary_weight, binary_bias)

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

class SimpleCNN(pl.LightningModule):

    def __init__(self, input_dim, n_classes, binarize = False, outpath = "."):
        super().__init__()
        # mnist images are (1, 28, 28) (channels, width, height) 
        if binarize:
            self.conv1 = BinaryConv2d(1, 32, 3, 1)
            self.bn_1 = nn.BatchNorm2d(32)
            self.activation_1 = BinaryTanh()
            self.pool_1 = nn.MaxPool2d(2)

            self.conv2 = BinaryConv2d(32, 32, 3, 1)
            self.bn_2 = nn.BatchNorm2d(32)
            self.activation_2 = BinaryTanh()
            self.pool_2 = nn.MaxPool2d(2)

            self.fc_1 = BinaryLinear(32 * 5 * 5, 32)
            self.bn = nn.BatchNorm1d(32)
            self.activation = BinaryTanh()
            self.out = BinaryLinear(32, 10)
        else:
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.bn_1 = nn.BatchNorm2d(32)
            self.activation_1 = nn.ReLU()
            self.pool_1 = nn.MaxPool2d(2)

            self.conv2 = nn.Conv2d(32, 32, 3, 1)
            self.bn_2 = nn.BatchNorm2d(32)
            self.activation_2 = nn.ReLU()
            self.pool_2 = nn.MaxPool2d(2)

            self.fc_1 = torch.nn.Linear(32 * 5 * 5, 32)
            self.bn = nn.BatchNorm1d(32)
            self.activation = nn.ReLU()
            self.out = torch.nn.Linear(32, 10)
        self.outpath = outpath
        self.input_dim = input_dim
        self.n_classes = n_classes
        
    def forward(self, x):
        #batch_size = x.shape[0]
        x = x.view((-1, 1, 28, 28))

        x = self.conv1(x)
        x = self.bn_1(x)
        x = self.activation_1(x)
        x = self.pool_1(x)

        x = self.conv2(x)
        x = self.bn_2(x)
        x = self.activation_2(x)
        x = self.pool_2(x)

        #x = x.view(batch_size, -1)
        #x = torch.nn.functional.flatten(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.out(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
        #return None

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        if loss is not None:
            self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        if loss is not None:
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
        #return os.path.join(out_path,model_name+".onnx")
         
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
        torch.onnx.export(model, dummy_x,onnx_path, training=torch.onnx.TrainingMode.PRESERVE, export_params=True,opset_version=12, do_constant_folding=False, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
        
        return onnx_path

def main():
    parser = argparse.ArgumentParser(description='Benchmark various CNN optimizations on the MNIST / Fashion dataset.')
    parser.add_argument('--outpath', required=True, help='Folder where data should written to.')
    parser.add_argument('--modelname', required=False, default="model", help='Modelname')
    parser.add_argument('--split','-s', required=False, default=0.2, type=float, help='Test/Train split.')
    parser.add_argument('--dataset','-d', required=True, help='Dataset to to be downloaded and used. Currently supported are {mnist, fashion}.')
    parser.add_argument("--binarize", "-b", required=False, action='store_true', help="Trains a binarized neural network if true.")
    args = parser.parse_args()

    if args.dataset in ["mnist", "fashion"]:
        n_features, n_classes = 28*28,10
    else:
        print("Only {eeg, magic, mnist, fashion} is supported for the --dataset/-d argument but you supplied {}.".format(args.dataset))
        sys.exit(1)

    model = SimpleCNN(n_features, n_classes, args.binarize, args.outpath)

    implementations = [ 
        ("NHWC",{}) 
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