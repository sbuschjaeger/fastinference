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

    def __init__(self, binarize = False):
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

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view((batch_size, 1, 28, 28))

        x = self.conv1(x)
        x = self.bn_1(x)
        x = self.activation_1(x)
        x = self.pool_1(x)

        x = self.conv2(x)
        x = self.bn_2(x)
        x = self.activation_2(x)
        x = self.pool_2(x)

        x = x.view(batch_size, -1)
        x = self.fc_1(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.out(x)
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
            print(model._modules[name].weight.data)
        
        # if isinstance(m, nn.BatchNorm2d):
        #     layer_new = WrappedBatchNorm(m)
        #     model._modules[name] = layer_new

    return model

def eval_model(model, x_train, y_train, x_test, y_test, out_path, name):
    print("Fitting {}".format(name))

    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    train_dataloader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=64)
    val_loader = None #DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=64)

    trainer = pl.Trainer(max_epochs = 1, default_root_dir = out_path, progress_bar_refresh_rate = 1)
    trainer.fit(model, train_dataloader, val_loader)
    model.eval() 
    
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
    print("Model accuracy is {}".format(djson["accuracy"]))
    # print("batch-latency: {}".format(djson["batch-latency"]))
    # print("single-latency: {}".format(djson["single-latency"]))

    with open(os.path.join(out_path, name + ".json"), "w") as outfile:  
        json.dump(djson, outfile) #, cls=NumpyEncoder

    if not (name.endswith("ONNX") or name.endswith("onnx")):
        name += ".onnx"

    print("Exporting {} to {}".format(name,out_path))
    # Export the model. Since torch traces the model we should put it in eval mode to 
    # prevent any updates to it (yes I know it sounds weird, but this way I sometimes ended
    # up with float weights after sanatizing -.-) 
    model = sanatize_onnx(model)

    print(model)
    print(model._modules["conv1"].weight.data)

    # https://github.com/pytorch/pytorch/issues/49229
    # set torch.onnx.TrainingMode.PRESERVE
    torch.onnx.export(model, dummy_x,os.path.join(out_path,name), training=torch.onnx.TrainingMode.PRESERVE, export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})

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
    
    model = SimpleCNN(args.binarize)
    eval_model(model, x_train, y_train, x_test, y_test, args.out, args.name)
    print("")

if __name__ == '__main__':

    main()

