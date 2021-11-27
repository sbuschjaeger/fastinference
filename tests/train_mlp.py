#!/usr/bin/env python3

import pandas as pd
import os
import argparse
from sklearn.metrics import accuracy_score

import datetime
import json
import onnx
import torch.onnx
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.autograd import Function

#from fastinference.Helper import NumpyEncoder

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

    def __init__(self, input_dim, n_classes, binarize = False):
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
        return self.forward(torch.from_numpy(X).float()).argmax(axis=1)   

    def on_epoch_start(self):
        print('\n')

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

def eval_model(model, x_train, y_train, x_test, y_test, out_path, name):
    print("Fitting {}".format(name))

    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()

    train_dataloader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=64)
    val_loader = None #DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=64)

    trainer = pl.Trainer(max_epochs = 1, default_root_dir = out_path, progress_bar_refresh_rate = 0)
    trainer.fit(model, train_dataloader, val_loader)
    model.eval() # This is to make sure that model is removed from the training state to avoid errors
    
    start_time = datetime.datetime.now()
    preds = model.predict(x_test)
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    batch_time = time_diff.total_seconds() * 1000
    accuracy = accuracy_score(y_test, preds)*100.0

    dummy_x = torch.randn(1, x_train.shape[1], requires_grad=False)
    start_time = datetime.datetime.now()
    preds = []
    for _ in range(x_test.shape[0]):
        preds.append(model.forward(dummy_x))
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    single_time = time_diff.total_seconds() * 1000

    djson = {
        "accuracy":accuracy,
        "name":name,
        "batch-latency": batch_time / x_test.shape[0],
        "single-latency": single_time / x_test.shape[0]
    }
    print("accuracy: {}".format(djson["accuracy"]))
    # print("batch-latency: {}".format(djson["batch-latency"]))
    # print("single-latency: {}".format(djson["single-latency"]))

    with open(os.path.join(out_path, name + ".json"), "w") as outfile:  
        json.dump(djson, outfile) #, cls=NumpyEncoder

    if not (name.endswith("ONNX") or name.endswith("onnx")):
        name += ".onnx"

    print("Exporting {} to {}".format(name,out_path))
    # Export the model, onnx file name:super_resolution.onnx
    model = sanatize_onnx(model)
    print(model)
    torch.onnx.export(model,dummy_x,os.path.join(out_path,name), export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})

    onnx_model = onnx.load(os.path.join(out_path,name))
    onnx.checker.check_model(onnx_model)


def main():
    parser = argparse.ArgumentParser(description='Train MLPs on the supplied data. This script assumes that each supplied training / testing CSV has a unique column called `label` which contains the labels.')
    parser.add_argument('--training', required=True, help='Filename of training data CSV-file')
    parser.add_argument('--testing', required=True, help='Filename of testing data CSV-file')
    parser.add_argument('--out', required=True, help='Folder where data should be written to.')
    parser.add_argument('--name', required=True, help='Modelname')
    parser.add_argument("--binarize", "-b", required=False, action='store_true', help="Trains a binarized neural network if true.")
    args = parser.parse_args()

    print("Loading training data")
    df = pd.read_csv(args.training)
    y_train = df["label"].to_numpy()
    x_train = df.drop(columns=["label"]).to_numpy()

    print("Loading testing data")
    df = pd.read_csv(args.testing)
    y_test = df["label"].to_numpy()
    x_test = df.drop(columns=["label"]).to_numpy()
    print("")
    
    n_classes = len(set(y_train) | set(y_test)) 
    input_dim = x_train.shape[1]

    model = SimpleMLP(input_dim, n_classes, args.binarize)
    eval_model(model, x_train, y_train, x_test, y_test, args.out, args.name)
    print("")

if __name__ == '__main__':
    main()
