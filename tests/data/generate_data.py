#!/usr/bin/env python3

import sys
import os
import pandas as pd

import argparse

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description='Simple tool to generate datasets of varying difficulty / number of classes / number of features.')
    parser.add_argument('--out', required=True, help='Filename where data should written to.')
    parser.add_argument('--nfeatures', required=False, type=int, default=2, help='Number of features')
    parser.add_argument('--nexamples', required=False, type=int, default=100, help='Number of examples')                        
    parser.add_argument('--nclasses', required=False, type=int, default=2, help='Number of classes')
    parser.add_argument('--difficulty', required=False, type=float, default=1.0, help='Difficulty level. Larger values are easier.')                        
    parser.add_argument('--split', required=False, default=0.33, type=float, help='If true, test and training data is exported.' )
    parser.add_argument('--float', required=False, default=False, action='store_true', help='True if features are float, else they are integer' )

    args = parser.parse_args()

    print("Generating data")
    X,y = make_classification(n_samples = args.nexamples, n_classes=args.nclasses, n_informative=args.nfeatures, class_sep=args.difficulty, n_features=args.nfeatures, n_redundant=0, n_repeated=0)

    if not args.float:
        X = (X*1000).astype(int)

    out_name = os.path.splitext(args.out)[0]

    print("Exporting data")
    if args.split > 0 and args.split < 1:
        XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size=args.split)
        dfTrain = pd.concat([pd.DataFrame(XTrain, columns=["f{}".format(i) for i in range(len(XTrain[0]))]), pd.DataFrame(YTrain,columns=["label"])], axis=1)
        dfTest = pd.concat([pd.DataFrame(XTest, columns=["f{}".format(i) for i in range(len(XTrain[0]))]), pd.DataFrame(YTest,columns=["label"])], axis=1)
        
        dfTrain.to_csv(os.path.join(out_name, "training.csv"), header=True, index=False)
        dfTest.to_csv(os.path.join(out_name, "testing.csv"), header=True, index=False)
    else:
        df = pd.concat([pd.DataFrame(X, columns=["f{}".format(i) for i in range(len(X[0]))]), pd.DataFrame(y, columns=["label"])], axis=1)
        df.to_csv(os.path.join(out_name, "data.csv"), header=True, index=False)


if __name__ == '__main__':
    main()
