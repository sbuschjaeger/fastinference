#!/usr/bin/env python3

import sys
import pandas as pd

import argparse

def main():
    # TODO: This script does not properly handle spaces in input parameters, e.g. 
    #       when "unsigned int" becomes ["unsigned", "int"]
    parser = argparse.ArgumentParser(description='Simple tool to convert datasets to static cpp-arrays')
    parser.add_argument('--file', required=True, help='CSV file to be read')
    parser.add_argument('--out', required=True,help='Filename where header should be written to.')
    parser.add_argument('--dtype', required=False, default="float", type=str, help='Data type of input data, e.g. float or int')
    parser.add_argument('--ltype', required=False, default="float", type=str,help='Data type of labels, e.g. float or int' )
    parser.add_argument('--nlines', required=False, default=None, type=int, help='Number of lines (=examples) from CSV used for exporting. If nlines = None (default) or nlines = 0 uses all available examples' )
    parser.add_argument('--shuffle', required=False, default=None, action='store_true', help='True if the dataset should be shuffled before exporting it.' )

    args = parser.parse_args()

    print("Reading ", args.file)
    df = pd.read_csv(args.file)

    # # Sometimes shell is weird and splits data / label types if they have a space in their name
    # # Thus we make sure to re-join them again
    # args.dtype = ''.join(args.dtype)
    # args.ltype = ''.join(args.ltype)

    if args.shuffle:
        df = df.sample(frac = 1) 

    Y = df["label"]
    X = df.drop(columns=["label"])

    if args.nlines is None or args.nlines == 0:
        X = X.head(n=args.nlines)
        Y = Y.head(n=args.nlines)
    
    X = X.to_numpy()
    Y = Y.to_numpy()

    print("Converting")
    x_str = ""
    y_str = ""
    for x,y in zip(X,Y):
        y_str += str(y) + ","
        for xi in x:
            x_str += str(xi) + ","

    x_str = x_str[:-1]
    y_str = y_str[:-1]

    print("Writing ", args.out)
    num_examples = len(X)
    num_features = len(X[0])
    num_classes = len(set(Y))
    with open(args.out, "w") as f:
        out_str = """
            #pragma once 
            static constexpr unsigned int NUM_EXAMPLES = {};
            static constexpr unsigned int NUM_FEATURES = {};
            static constexpr unsigned int NUM_CLASSES = {};
            static constexpr {} X[] = {{ {} }};
            static constexpr {} Y[] = {{ {} }};
            """.format(num_examples, num_features, num_classes, args.dtype, x_str, args.ltype, y_str)
        f.write(out_str)

if __name__ == '__main__':
    main()
