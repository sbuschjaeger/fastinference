#!/usr/bin/env python3

import itertools
import os
import pandas as pd
import argparse

from torch import optim
from test_utils import get_dataset, prepare_fastinference, run_experiment, make_hash, test_implementations

import fastinference.Loader

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Benchmark various tree optimizations on the supplied dataset.')
    parser.add_argument('--outpath', required=True, help='Folder where data should written to.')
    parser.add_argument('--dataset','-d', required=True, help='Dataset to to be downloaded and used. Currently supported are {magic, mnist, fashion, eeg}.')
    parser.add_argument('--split','-s', required=False, default=0.2, type=float, help='Test/Train split.')

    parser.add_argument('--modelname', required=False, default="model", help='Modelname')
    parser.add_argument('--nestimators', required=False, type=int, default=128,help='Number of trees in a random forest.')
    parser.add_argument('--maxdepth', required=False, type=int, default=20,help='Maximum tree-depth for decision trees and random forest.')
    args = parser.parse_args()
    
    performance = []

    XTrain, YTrain, _, _ = get_dataset(args.dataset,split=args.split)

    implementations = [ 
        ("ifelse",{"label_type":"double", "feature_type":"double"}), 
        ("native",{"label_type":"double"})] + [
        ("ifelse",{"kernel_type":"path", "kernel_budget":b}) for b in [128]] + [
        ("ifelse",{"kernel_type":"node", "kernel_budget":b}) for b in [128]] + [
        ("native", {"reorder_nodes":True, "set_size":s}) for s in [8]
    ]

    if args.nestimators <= 1:
        model = DecisionTreeClassifier(max_depth=args.maxdepth)
        base_optimizers = [
            ([None], [{}]),
            (["quantize"],[{"quantize_splits":"rounding", "quantize_leafs":1000}]),
            (["quantize"],[{"quantize_leafs":1000, "quantize_splits":1000}]),
        ]
        ensemble_optimizers = [
            ([None], [{}])
        ]
    else:
        model = RandomForestClassifier(n_estimators=args.nestimators, max_depth=args.maxdepth, max_leaf_nodes=512)
        base_optimizers = [
            ([None], [{}]),
        ]

        ensemble_optimizers = [
            ([None], [{}]),
            (["quantize"],[{"quantize_splits":"rounding", "quantize_leafs":1000}]),
            (["leaf-refinement"], [{"X":XTrain, "Y":YTrain, "epochs":1, "optimizer":"adam", "verbose":True}]),
            (["weight-refinement"], [{"X":XTrain, "Y":YTrain, "epochs":1, "optimizer":"sgd", "verbose":True}])
        ]

    performance = test_implementations(model = model, dataset= args.dataset, split = args.split, implementations = implementations, base_optimizers = base_optimizers, ensemble_optimizers=ensemble_optimizers, out_path = args.outpath, model_name = args.modelname)

    df = pd.DataFrame(performance)
    with pd.option_context('display.max_rows', None): 
        print(df)

if __name__ == '__main__':
    main()
    
