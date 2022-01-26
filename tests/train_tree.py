#!/usr/bin/env python3

import sys
import os
from joblib import dump, load
import pandas as pd
import pathlib

import argparse

import fastinference.Loader

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def eval_model(model, x_train, y_train, x_test, y_test, out_path, name):
    print("Fitting {}".format(name))
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))*100.0

    print("Exporting {}".format(name))
    fi_model = fastinference.Loader.model_from_sklearn(model, name, accuracy=accuracy)
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print("Exporting {} to {}".format(name,out_path))

    # print(fi_model.models)    
    # print("out_path: {}".format(out_path))
    fastinference.Loader.model_to_json(fi_model,out_path, name)
    dump(model, os.path.join(out_path,"{}.joblib".format(name)))

def main():
    parser = argparse.ArgumentParser(description='Train linear models on supplied data. This script assumes that each supplied training / testing CSV has a unique column called `label` which contains the labels.')
    parser.add_argument('--training', required=True,help='Filename of training data CSV-file')
    parser.add_argument('--testing', required=True,help='Filename of testing data CSV-file')
    parser.add_argument('--out', required=True,help='Folder where data should written to.')
    parser.add_argument('--name', required=True, help='Modelname')
    parser.add_argument('--nestimators', required=False, type=int, default=8,help='Number of  in a random forest.')
    parser.add_argument('--maxdepth', required=False, type=int, default=20,help='Maximum tree-depth for decision  and random forest.')
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

    if args.nestimators <= 1:
        tree = DecisionTreeClassifier(max_depth=args.maxdepth)
        eval_model(tree, x_train, y_train, x_test, y_test, args.out, args.name)
        print("")
    else:
        rf = RandomForestClassifier(n_estimators=args.nestimators,max_depth=args.maxdepth)
        eval_model(rf, x_train, y_train, x_test, y_test, args.out, args.name)
        print("")

if __name__ == '__main__':
    main()

# print(df.head(n=5))
# print(x)
# print(y)