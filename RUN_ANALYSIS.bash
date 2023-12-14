#!/bin/env bash

# This code will reproduce the main figures and tables from the paper "Development of prediction models to identify hotspots of schistosomiasis in endemic regions to guide mass drug administration" Singer et al.
# The code is written in Python 3.7.3 and requires the following packages: numpy, pandas, matplotlib, sklearn, scipy, and xgboost.
# It requires the data file all_hotspots_prediction_data.csv.
# The code is run from the command line using the following command: bash RUN_ANALYSIS.bash

# select analysis type (default 70:30 train-test split with cross-validation, for 6:2:2 train-validation-test split validation_approach="Fixed")
folder=""
validation_approach=""

# split training and test data, with random seed 220516
mkdir "$folder""Data/""$validation_approach""Train_Data"
mkdir "$folder""Data/""$validation_approach""Test_Data"
mkdir "$folder""Outputs/""$validation_approach""Trained_Models"
python3 Analysis/Data_Processing/train_test_separation.py "$folder" 220516 "$validation_approach"

# generate models and outcome tables for the main analysis, the analysis excluding secondary data (var="alt_"), and the analysis with prevalence corrected for diagnostic sensitivity
for var in "" "alt_" "true_"
do
	# train models, with cross-validation summary figures
	python3 Analysis/Data_Processing/scaling.py "$folder" "$var" "$validation_approach"
	for model in Analysis/Models/linear_regression.py Analysis/Models/logistic_regression.py Analysis/Models/elastic_net.py Analysis/Models/random_forest.py Analysis/Models/XGBoost.py Analysis/Models/SVM.py Analysis/Models/ANN.py
	do 
		python3 $model "$folder" "$var" "$validation_approach" 1
	done
	python3 ensemble.py "$folder" "$var" "$validation_approach"
done

# evaluate (test/validate) models, with CIs
for var in "" "alt_"
do
	echo "Model,Trainset,Outcome,Metric,Score,Lower,Upper" >> "$folder""$var""$val""$validation_approach""evaluate.csv"
	for model in EnsembleRegressor LinearRegression ElasticNetLinear RandomForestRegressor GradientBoostingRegressor SupportVectorRegressor MultilayerPerceptronRegressor
	do
		for trainset in NIG KEN TAN Sm Sh all_NIG all_COTKEN all_KENTAN all_TANCOT
		do
			python3 Analysis/evaluate.py $model $trainset "Prevalence" 0.95 "$folder" "$var" "$validation_approach" >> "$folder""$var""$validation_approach""evaluate.csv"
		done
	done
	for model in EnsembleClassifier LogisticRegression ElasticNetLogistic RandomForestClassifier GradientBoostingClassifier SupportVectorClassifier MultilayerPerceptronClassifier
	do 
		for trainset in NIG KEN TAN Sm Sh all_NIG all_COTKEN all_KENTAN all_TANCOT
		do
			for outcome in "Prevalence" "Prevalence Relative" "Prevalence Intensity"
			do
				python3 Analysis/evaluate.py $model $trainset "$outcome" 0.95 "$folder" "$var" "$validation_approach" >> "$folder""$var""$validation_approach""evaluate.csv"
			done
		done
	done
done

# evaluate (test/validate) models, with CIs, with 'true' outcomes
echo "Model,Trainset,Outcome,Metric,Score,Lower,Upper" >> "$folder""$var""$val""$validation_approach""evaluate.csv"
for model in EnsembleRegressor LinearRegression ElasticNetLinear RandomForestRegressor GradientBoostingRegressor SupportVectorRegressor MultilayerPerceptronRegressor
do
	for trainset in NIG KEN TAN Sm Sh all_NIG all_COTKEN all_KENTAN all_TANCOT
	do
		python3 Analysis/evaluate.py $model $trainset "True Prevalence" 0.95 "$folder" "$var" "$validation_approach" >> "$folder""$var""$validation_approach""evaluate.csv"
	done
done
for model in EnsembleClassifier LogisticRegression ElasticNetLogistic RandomForestClassifier GradientBoostingClassifier SupportVectorClassifier MultilayerPerceptronClassifier
do 
	for trainset in NIG KEN TAN Sm Sh all_NIG all_COTKEN all_KENTAN all_TANCOT
	do
		for outcome in "True Prevalence" "True Prevalence Relative" "True Prevalence Intensity"
		do
			python3 Analysis/evaluate.py $model $trainset "$outcome" 0.95 "$folder" "$var" "$validation_approach" >> "$folder""$var""$validation_approach""evaluate.csv"
		done
	done
done

# tables of outcomes (coresponding to SI tables)
for var in "" "alt_" "true_"
do
	# generate tables of outcomes with CIs
	for metric in Accuracy Sensitivity Specificity Balanced FPR PPV NPV
	do
		python3 score_tables.py "$folder""$var""$validation_approach""evaluate.csv" $metric 1 "$var" >> "$folder""Figures/""$var""$validation_approach""$metric""_table.tex"
	done
	# F1 and AUC-ROC without CIs
	for metric in F1 AUC
	do
		python3 score_tables.py "$folder""$var""$validation_approach""evaluate.csv" $metric 0 "$var" >> "$folder""Figures/""$var""$validation_approach""$metric""_table.tex"
	done
done

# generate tex code for table 2
python3 best_tables.py "$folder""$validation_approach""evaluate.csv" 1 "" >> "$folder""Figures/""$validation_approach""best_table.tex"

# generate tex code for table 3
python3 difference_tables.py "$folder""$validation_approach""evaluate.csv" "$folder""alt_""$validation_approach""evaluate.csv" >> "$folder""Figures/""$validation_approach""diff_table.tex"

# generate figure 2 (folder+validation_approach+'task_heatplots_BalancedEnsembleClassifier.png')
python3 task_heatplots.py "$folder""$validation_approach""evaluate.csv" Balanced EnsembleClassifier "$folder" "" "$validation_approach"

# generate figure 3 (hotspot_incidence_recommended_line_plots.png)
python3 hotspot_incidence_recommended_line_plots.py evaluate.csv "$folder"