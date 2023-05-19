#!/bin/env bash

mkdir "/Train_Data"
mkdir "/Test_Data"
mkdir "/Trained_Models"
python3 train_test_separation.py "" 0
python3 scaling.py "" ""
for model in linear_regression.py logistic_regression.py random_forest.py boosted_trees.py elastic_net.py
do 
	python3 $model "" "" ""
done
python3 ensemble.py "" "" ""
folder=""
export folder
bash run_evaluate.bash
