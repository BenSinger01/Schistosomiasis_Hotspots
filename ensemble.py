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

folder, var, val, validation_approach = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
clas_targets=["Prevalence Hotspot","Prevalence Relative Hotspot","Prevalence Intensity Hotspot"]
reg_targets=["Prevalence Outcome"]
if var == "true_":
	clas_targets=["True Prevalence Hotspot","True Prevalence Relative Hotspot","True Prevalence Intensity Hotspot","Prevalence Hotspot","Prevalence Relative Hotspot","Prevalence Intensity Hotspot"]
	reg_targets = ["True Prevalence Outcome","Prevalence Outcome"]

variable_func = eval('variable_func.'+var+'variable_func')


for trainset in ["NIG","KEN","TAN","Sm","Sh","all_NIG"
,"all_COTKEN","all_KENTAN","all_TANCOT"]:
	data = pd.read_csv(folder+validation_approach+"Train_Data/"+trainset+".csv")
	variables = variable_func(trainset)
	X_unscaled = data[variables]	
	with open(folder+validation_approach+"Train_Data/"+trainset+"_"+var+"scaler.pickle",'rb') as pklfile:
		scaler = pkl.load(pklfile)
	X = scaler.transform(X_unscaled)
	for target in reg_targets:
		reg_models = ['LinearRegression','ElasticNetLinear'
		,'RandomForestRegressor','GradientBoostingRegressor'
		,'SupportVectorRegressor','MultilayerPerceptronRegressor']
		estimators = []
		for model in reg_models:
			with open(folder+var+val+validation_approach+"Trained_Models/"+model+"_"
				+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'rb') as rfile:
				reg_fit = pkl.load(rfile)
				estimators.append((model,reg_fit))
		vot_reg = ensemble.VotingRegressor(estimators=estimators)
		y = np.array(data[target])
		vot_reg_fit = vot_reg.fit(X,y)
		with open(folder+var+val+validation_approach+"Trained_Models/EnsembleRegressor_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(vot_reg_fit,wfile)
	for target in clas_targets:
		clas_models = ['LogisticRegression','ElasticNetLogistic'
		,'RandomForestClassifier','GradientBoostingClassifier'
		,'SupportVectorClassifier','MultilayerPerceptronClassifier']
		estimators = []
		for model in clas_models:
			with open(folder+var+val+validation_approach+"Trained_Models/"+model+"_"
				+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'rb') as rfile:
				reg_fit = pkl.load(rfile)
				estimators.append((model,reg_fit))
		vot_reg = ensemble.VotingClassifier(estimators=estimators,voting='hard')
		y = np.array(data[target])
		vot_reg_fit = vot_reg.fit(X,y)
		with open(folder+var+val+validation_approach+"Trained_Models/EnsembleClassifier_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(vot_reg_fit,wfile)