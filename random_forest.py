import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn import ensemble
import pickle as pkl
import csv
import sys
import variable_func
import validation_score

folder, var, val = sys.argv[1], sys.argv[2], sys.argv[3]

variable_func = eval('variable_func.'+var+'variable_func')
validation_score = eval('validation_score.'+val+'validation_score')

np.random.seed(220517)


# test all sizes of feature subset, and try different bootstrap sizes
search_params = {
"max_features" : np.linspace(1/17,1,17)
,"max_samples" : np.linspace(.1,.99,10)
}

for trainset in ["NIG","KEN","TAN","Sm","Sh","all_NIG"
,"all_COTKEN","all_KENTAN","all_TANCOT"]:
	for target in ["Prevalence Outcome","Intensity Outcome","Relative Outcome"]:
		data = pd.read_csv(folder+"Train_Data/"+trainset+".csv")
		country_indicator =\
		 [item==data["Country_Code"].iloc[0] for item in data["Country_Code"]]
		variables = variable_func(trainset)
		X_unscaled = data[variables]
		y = np.array(data[target])
		with open(folder+"Train_Data/"+trainset+"_"+var+"scaler.pickle",'rb') as pklfile:
			scaler = pkl.load(pklfile)
		X = scaler.transform(X_unscaled)
		scorer = metrics.make_scorer(validation_score,target=target)
		reg = ensemble.RandomForestRegressor(n_estimators=300)
		if len(trainset)>7:
			search = model_selection.GridSearchCV(reg,param_grid=search_params
				,scoring=scorer
				,cv=model_selection.LeaveOneGroupOut())
			search_fit = search.fit(X,y
				,groups=country_indicator)
		else:
			search = model_selection.GridSearchCV(reg,param_grid=search_params
				,scoring=scorer
				,cv=5)
			search_fit = search.fit(X,y)
		reg_fit = search_fit.best_estimator_
		with open(folder+var+val+"Trained_Models/RandomForestRegressor_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(reg_fit,wfile)
		with open(folder+var+val+"Trained_Models/RandomForestRegressor_"
			+trainset+"_"+target.replace(' ','_')+"_search.pickle",'wb') as wfile:
			pkl.dump(search_fit,wfile)

for trainset in ["NIG","KEN","TAN","Sm","Sh","all_NIG"
,"all_COTKEN","all_KENTAN","all_TANCOT"]:
	for target in ["Prevalence Hotspot","Intensity Hotspot","Relative Hotspot"]:
		data = pd.read_csv(folder+"Train_Data/"+trainset+".csv")
		country_indicator =\
		 [item==data["Country_Code"].iloc[0] for item in data["Country_Code"]]
		variables = variable_func(trainset)
		X_unscaled = data[variables]
		y = np.array(data[target])
		with open(folder+"Train_Data/"+trainset+"_"+var+"scaler.pickle",'rb') as pklfile:
			scaler = pkl.load(pklfile)
		X = scaler.transform(X_unscaled)
		clas = ensemble.RandomForestClassifier(n_estimators=300)
		if len(trainset)>7:
			search = model_selection.GridSearchCV(clas,param_grid=search_params
				,scoring=val+"accuracy"
				,cv=model_selection.LeaveOneGroupOut())
			search_fit = search.fit(X,y
				,groups=country_indicator)
		else:
			search = model_selection.GridSearchCV(clas,param_grid=search_params
				,scoring=val+"accuracy"
				,cv=5)
			search_fit = search.fit(X,y)
		clas_fit = search_fit.best_estimator_
		with open(folder+var+val+"Trained_Models/RandomForestClassifier_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(clas_fit,wfile)
		with open(folder+var+val+"Trained_Models/RandomForestClassifier_"
			+trainset+"_"+target.replace(' ','_')+"_search.pickle",'wb') as wfile:
			pkl.dump(search_fit,wfile)