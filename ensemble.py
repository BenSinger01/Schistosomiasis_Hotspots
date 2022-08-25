import pickle as pkl
import csv
import pandas as pd
import numpy as np
from sklearn import ensemble
import sys
import variable_func
import validation_score

folder, var, val = sys.argv[1], sys.argv[2], sys.argv[3]

variable_func = eval('variable_func.'+var+'variable_func')

for trainset in ["NIG","KEN","TAN","Sm","Sh","all_NIG"
,"all_COTKEN","all_KENTAN","all_TANCOT"]:
	for target in ["Prevalence Outcome","Intensity Outcome","Relative Outcome"]:
		estimators = []
		for model in ['LinearRegression','ElasticNetLinear'
		,'RandomForestRegressor','GradientBoostingRegressor']:
			with open(folder+var+val+"Trained_Models/"+model+"_"
				+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'rb') as rfile:
				reg_fit = pkl.load(rfile)
				estimators.append((model,reg_fit))
		vot_reg = ensemble.VotingRegressor(estimators=estimators)

		data = pd.read_csv(folder+"Train_Data/"+trainset+".csv")
		variables = variable_func(trainset)
		X_unscaled = data[variables]
		y = np.array(data[target])
		with open(folder+"Train_Data/"+trainset+"_"+var+"scaler.pickle",'rb') as pklfile:
			scaler = pkl.load(pklfile)
		X = scaler.transform(X_unscaled)

		vot_reg_fit = vot_reg.fit(X,y)
		with open(folder+var+val+"Trained_Models/EnsembleRegressor_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(vot_reg_fit,wfile)

for trainset in ["NIG","KEN","TAN","Sm","Sh","all_NIG"
,"all_COTKEN","all_KENTAN","all_TANCOT"]:
	for target in ["Prevalence Hotspot","Intensity Hotspot","Relative Hotspot"]:
		estimators = []
		for model in ['LogisticRegression','ElasticNetLogistic'
		,'RandomForestClassifier','GradientBoostingClassifier']:
			with open(folder+var+val+"Trained_Models/"+model+"_"
				+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'rb') as rfile:
				reg_fit = pkl.load(rfile)
				estimators.append((model,reg_fit))
		vot_reg = ensemble.VotingClassifier(estimators=estimators,voting='soft')

		data = pd.read_csv(folder+"Train_Data/"+trainset+".csv")
		variables = variable_func(trainset)
		X_unscaled = data[variables]
		y = np.array(data[target])
		with open(folder+"Train_Data/"+trainset+"_"+var+"scaler.pickle",'rb') as pklfile:
			scaler = pkl.load(pklfile)
		X = scaler.transform(X_unscaled)

		vot_reg_fit = vot_reg.fit(X,y)
		with open(folder+var+val+"Trained_Models/EnsembleClassifier_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(vot_reg_fit,wfile)