import pickle as pkl
import csv
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
import sys
import variable_func
import validation_score

np.random.seed(230803)

# load control variables from command line
folder, var, val, validation_approach = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
clas_targets=["Prevalence Hotspot","Prevalence Relative Hotspot","Prevalence Intensity Hotspot"]
reg_targets=["Prevalence Outcome"]
if var == "true_":
	clas_targets=["True Prevalence Hotspot","True Prevalence Relative Hotspot","True Prevalence Intensity Hotspot","Prevalence Hotspot","Prevalence Relative Hotspot","Prevalence Intensity Hotspot"]
	reg_targets = ["True Prevalence Outcome","Prevalence Outcome"]

variable_func = eval('variable_func.'+var+'variable_func')

# put together ensemble models
# iterate over train sets (and corresponding test sets)
for trainset in ["NIG","KEN","TAN","Sm","Sh","all_NIG"
,"all_COTKEN","all_KENTAN","all_TANCOT"]:

	# load data and variables
	data = pd.read_csv(folder+"Data/"+validation_approach+"Train_Data/"+trainset+".csv")
	variables = variable_func(trainset)
	X_unscaled = data[variables]

	# scale variables
	with open(folder+"Data/"+validation_approach+"Train_Data/"+trainset+"_"+var+"scaler.pickle",'rb') as pklfile:
		scaler = pkl.load(pklfile)
	X = scaler.transform(X_unscaled)

	# ensemble regression model for each hotspot definition
	for target in reg_targets:

		# load each regression model type
		reg_models = ['LinearRegression','ElasticNetLinear'
		,'RandomForestRegressor','GradientBoostingRegressor'
		,'SupportVectorRegressor','MultilayerPerceptronRegressor']
		estimators = []
		for model in reg_models:
			with open(folder+"Outputs"+var+val+validation_approach+"Trained_Models/"+model+"_"
				+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'rb') as rfile:
				reg_fit = pkl.load(rfile)
				estimators.append((model,reg_fit))

		# create and fit ensemble model
		vot_reg = ensemble.VotingRegressor(estimators=estimators)
		y = np.array(data[target])
		vot_reg_fit = vot_reg.fit(X,y)

		# save pickled model
		with open(folder+"Outputs"+var+val+validation_approach+"Trained_Models/EnsembleRegressor_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(vot_reg_fit,wfile)

	# ensemble classification model for each hotspot definition
	for target in clas_targets:

		# load each classification model type
		clas_models = ['LogisticRegression','ElasticNetLogistic'
		,'RandomForestClassifier','GradientBoostingClassifier'
		,'SupportVectorClassifier','MultilayerPerceptronClassifier']
		estimators = []
		for model in clas_models:
			with open(folder+"Outputs"+var+val+validation_approach+"Trained_Models/"+model+"_"
				+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'rb') as rfile:
				reg_fit = pkl.load(rfile)
				estimators.append((model,reg_fit))

		# create and fit ensemble model
		vot_reg = ensemble.VotingClassifier(estimators=estimators,voting='hard')
		y = np.array(data[target])
		vot_reg_fit = vot_reg.fit(X,y)

		# save pickled model
		with open(folder+"Outputs"+var+val+validation_approach+"Trained_Models/EnsembleClassifier_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(vot_reg_fit,wfile)