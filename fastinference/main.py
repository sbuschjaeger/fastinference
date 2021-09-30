#!/usr/bin/env python3

import argparse,os
from fastinference.models.Ensemble import Ensemble
import logging
import json
import sys

from Util import dynamic_import 
#from Loader import model_from_file
import Loader
# from models.nn.NeuralNet import NeuralNet
# from models.Linear import Linear

def parse_unknown(unknown):
    '''
    Parse every unknown argument. This assumes lists: 
        - [..., '--feature_type', 'float', '--other_type', 'int', ...] will be parsed to {"feature_type": "float"}
        - [..., '--feature_type', 'float', 'int', ...] will be parsed to {"feature_type": ["float", "int"]}
        - [..., '--feature_type', '--other_type', 'int', ... ] will be parsed to {"feature_type": True}
    '''
    d = {}
    cur_key = None
    for u in unknown:
        # Check if there is a new key for the dictionary and init list. Otherwise append to the current flag
        if "--" in u:
            cur_key = u[2:]
            d[cur_key] = []
        else:
            d[cur_key].append(u)
    
    # Flatten the dictionary appropriatly
    for key in d:
        if isinstance(d[key], list):
            if len(d[key]) == 0:
                d[key] = True
            elif len(d[key]) == 1:
                d[key] = d[key][0]
            
    return d

def main():
    '''
    The main entry point of fastinference :-)
    '''
    parser = argparse.ArgumentParser(description='Fastinference is awesome!.')
    parser.add_argument('--model', required=True, help='Specify a model file. Currently supported are neural networks (ONNX model), linear models, decision trees, discriminant analysie and ensembles of those. Neural Networks should be given as ONNX files, whereas any other models (including ensembles of Neural Networks) should be given as JSON files. ')
    parser.add_argument('--implementation', required=True, help = "Target implementation.", type=str)
    parser.add_argument('--out_name', required=False, default="model", type=str, help='Filename under which the generated code should be stored.')
    parser.add_argument('--out_path', required=False, default=".", type=str, help='Folder under which the generated code should be stored.')
    parser.add_argument('-v', '--verbose', action="store_const", const=logging.INFO, default=logging.WARNING, help="Be verbose.")

    args, unknown = parser.parse_known_args()
    unknown = parse_unknown(unknown)

    logging.basicConfig(level=args.verbose)
    loaded_model = Loader.model_from_file(args.model)

    to_implementation = dynamic_import("templates.{}.{}.implement".format(loaded_model.category,args.implementation), "to_implementation")
    os.makedirs(args.out_path, exist_ok = True)
    to_implementation(loaded_model, args.out_path, args.out_name, weight = 1.0, **unknown)

if __name__ == '__main__':
    # sys.argv = ['fastinference/main.py', '--model', '/tmp/fastinference/SimpleCNN/SimpleCNN.onnx', '--feature_type', 'int', '--out_path', '/tmp/fastinference/SimpleCNN', '--out_name', 'model', '--implementation', 'cpp.binary']
    print(sys.argv)
    main()
    # python3 -m fastinference --model /tmp/fastinference/nn/cpp/binary/SimpleMLP.onnx --feature_type int --implementation cpp.NHWC --out_path /tmp/fastinference/nn/cpp/binary/ --out_name "model"