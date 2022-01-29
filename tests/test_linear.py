#!/usr/bin/env python3

import pandas as pd
import argparse
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from test_utils import test_implementations

from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser(description='Benchmark various tree optimizations on the supplied dataset.')
    parser.add_argument('--outpath', required=True, help='Folder where data should written to.')
    parser.add_argument('--dataset','-d', required=True, help='Dataset to to be downloaded and used. Currently supported are {magic, mnist, fashion, eeg}.')
    parser.add_argument('--split','-s', required=False, default=0.2, type=float, help='Test/Train split.')
    parser.add_argument('--type','-t', required=False, default="linear",help='Type of model to be tested. Can be {linear, quadratic}')
    parser.add_argument('--modelname', required=False, default="model", help='Modelname')
    # parser.add_argument('--nestimators', required=False, type=int, default=128,help='Number of trees in a random forest.')
    args = parser.parse_args()

    ensemble_optimizers = [
        ([None], [{}])
    ]
    if args.type == "linear":
        model = RidgeClassifier(alpha=1.0)
    else:
        model = QuadraticDiscriminantAnalysis()
    # else:
    #     if args.type == "linear":
    #         model = BaggingClassifier(RidgeClassifier(alpha=1.0),n_estimators=args.nestimators)
    #         ensemble_optimizers = [
    #             #([None], [{}]),
    #             (["merge-linear"], [{}]),
    #         ]
    #     else:
    #         model = BaggingClassifier(QuadraticDiscriminantAnalysis(),n_estimators=args.nestimators)
    #         ensemble_optimizers = []

    if args.type == "linear":
        implementations = [ 
            ("native",{}), 
            #("unroll",{})
        ] 
    else:
        implementations = [ 
            ("native",{}),
        ] 

    optimizers = [
        ([None], [{}])
    ]

    performance = test_implementations(model = model, dataset= args.dataset, split = args.split, implementations = implementations, base_optimizers = optimizers, out_path = args.outpath, model_name = args.modelname)
    df = pd.DataFrame(performance)
    print(df)


if __name__ == '__main__':
    main()
    
