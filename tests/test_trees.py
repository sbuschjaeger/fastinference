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
    
    print("Loading {}".format(args.dataset))
    XTrain, YTrain, XTest, YTest = get_dataset(args.dataset,args.outpath,args.split)
    
    dfTest = pd.concat([pd.DataFrame(XTest, columns=["f{}".format(i) for i in range(len(XTrain[0]))]), pd.DataFrame(YTest,columns=["label"])], axis=1)
    path_to_testfile = os.path.join(args.outpath, "testing.csv")
    dfTest.to_csv(path_to_testfile, header=True, index=False)

    performance = []

    if args.nestimators <= 1:
        model = DecisionTreeClassifier(max_depth=args.maxdepth)
    else:
        model = RandomForestClassifier(n_estimators=args.nestimators, max_depth=args.maxdepth, max_leaf_nodes=512)
    
    print("Storing model.")
    model.fit(XTrain, YTrain)
    acc = accuracy_score(model.predict(XTest), YTest)*100.0
    fimodel = fastinference.Loader.model_from_sklearn(model, name = args.modelname, accuracy = acc)
    fastinference.Loader.model_to_json(fimodel, os.path.join(args.outpath), file_name=args.modelname)

    implementations = [ 
        ("ifelse",{}), 
        ("native",{})] + [
        ("ifelse",{"kernel_type":"path", "kernel_budget":b}) for b in [128]] + [
        ("ifelse",{"kernel_type":"node", "kernel_budget":b}) for b in [128]] + [
        ("native", {"reorder_nodes":True, "set_size":s}) for s in [8]
    ]

    optimizers = [
        ([None], [{}]),
        (["quantize"],[{"quantize_splits":"rounding", "quantize_leafs":"fixed", "quantize_factor":1000}]),
        (["quantize"],[{"quantize_leafs":"fixed", "quantize_factor":1000}])
    ]

    performance = test_implementations(model, args.dataset, args.split, implementations, optimizers, args.outpath, args.modelname)
    df = pd.DataFrame(performance)
    print(df)

    # for impl, opt in itertools.product(implementations, optimizers):
    #     name = args.modelname + "_" + make_hash(impl) + "_" + make_hash(opt)
    #     model_path = os.path.join(args.outpath, name)
    #     prepare_fastinference(model_path, args.outpath, name, path_to_testfile, implementation_type = impl[0], implementation_args = impl[1], base_optimizer = opt[0], base_optimizer_args = opt[1])

    #     performance.append(
    #         {
    #             "implementation":impl[0],
    #             "implementation_args":impl[1],
    #             "optimizer":opt[0],
    #             "optimizer_args":opt[1],
    #             **run_experiment(outpath, name, path_to_testfile,  model.n_classes_, 5)
    #         }
    #     )

    # prepare_fastinference(model,os.path.join(args.outpath, "ifelse"), "ifelse", path_to_testfile, "ifelse")
    # performance.append(run_experiment(os.path.join(args.outpath, "ifelse"), "ifelse", path_to_testfile, model.n_classes_, 5 ))

    # for m in ["individual_contribution", "individual_error", "reduced_error", "complementariness", "cluster_accuracy", "largest_mean_distance"]:
    #     name = "native_{}".format(m)
    #     out_path = os.path.join(args.outpath, name)
    #     prepare_fastinference(model,out_path, name, path_to_testfile, "native", {}, ensemble_optimizer="pruning", 
    #         ensemble_optimizer_args={
    #             "pruning_method":m, "n_estimators":10, "x_prune":XTrain, "y_prune":YTrain
    #         }
    #     )
    #     performance.append(run_experiment(out_path, name, path_to_testfile, model.n_classes_, 5))
    
    # name = "native_drep"
    # out_path = os.path.join(args.outpath, name)
    # prepare_fastinference(model,out_path, name, path_to_testfile, "native", {}, ensemble_optimizer="pruning", 
    #     ensemble_optimizer_args={
    #         "pruning_method":"drep", "n_estimators":10, "x_prune":XTrain, "y_prune":YTrain, "metric_options": {"rho": 0.25},
    #     }
    # )
    # performance.append(run_experiment(out_path, name, path_to_testfile,  model.n_classes_, 5))

    # name = "native_rf_small"
    # model = RandomForestClassifier(n_estimators=10,max_depth=args.maxdepth)
    # model.fit(XTrain, YTrain)
    # out_path = os.path.join(args.outpath, name)
    # prepare_fastinference(model,out_path, name, path_to_testfile, "native")
    # performance.append(run_experiment(out_path, name, path_to_testfile, model.n_classes_, 5))

    # from PyPruning.Papers import create_pruner
    # lf_pruner = create_pruner(
    #     "leaf_refinement", **{"batch_size" : 128, "epochs": 50, "step_size": 1e-1, "verbose":False, "loss":"mse"}
    # )
    # lf_pruner.prune(XTrain, YTrain, model.estimators_, model.classes_, model.n_classes_)
    # model.estimators_ = lf_pruner.estimators_
    # model.weights_ = lf_pruner.weights_

    # name = "native_lr"
    # out_path = os.path.join(args.outpath, name)
    # prepare_fastinference(model,out_path, name, path_to_testfile, "native")
    # performance.append(run_experiment(out_path, name, path_to_testfile, model.n_classes_, 5, ))
    # print("DONE")

    # performance.append(eval_fastinference(model,os.path.join(args.outpath, "ifelse_swap"), "ifelse_swap", args.testing, 5, "ifelse", optimizer="swap"))
    # for s in [1024, 2048, 4096]:
    #     performance.append(eval_fastinference(model,os.path.join(args.outpath, "ifelse_swap_path_{}".format(s)), "ifelse_swap_path_{}".format(s), args.testing, 5, "ifelse", optimizer="swap", implementation_args={"kernel_type":"path", "kernel_budget":s}))
    #     performance.append(eval_fastinference(model,os.path.join(args.outpath, "ifelse_swap_node_{}".format(s)), "ifelse_swap_node_{}".format(s), args.testing, 5, "ifelse", optimizer="swap", implementation_args={"kernel_type":"node", "kernel_budget":s}))

    # performance.append(eval_fastinference(model,os.path.join(args.outpath, "native"), "native", args.testing, 5, "native"))
    # for s in [2,4,6,8,10]:
    #     performance.append(eval_fastinference(model,os.path.join(args.outpath, "native_optimized_{}".format(s)), "native_optimized_{}".format(s), args.testing, 5, implementation_type="native", implementation_args= {"set_size":s}))
    #     performance.append(eval_fastinference(model,os.path.join(args.outpath, "native_optimized_full_{}".format(s)), "native_optimized_full_{}".format(s), args.testing, 5, implementation_type="native", implementation_args= {"set_size":s, "force_cacheline":True}))
    # performance.append(eval_treelite(model,os.path.join(args.outpath, "treelite"), args.modelname, args.testing, 5))
   
    # df = pd.DataFrame(performance)
    # print(df)

if __name__ == '__main__':
    main()
    
