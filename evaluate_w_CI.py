from sklearn import metrics
from sklearn import model_selection
from wilson_score_interval import interval
import pickle as pkl
import numpy as np
import pandas as pd
import variable_func
import csv
import sys

model, trainset, outcome, metric, p_value, folder, var, val =\
 sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6], sys.argv[7], sys.argv[8]

variable_func = eval('variable_func.'+var+'variable_func')

testset = ["NIG","KEN","TAN","Sm","Sh","all_MOZ","all_TAN","all_COT","all_KEN"]\
[["NIG","KEN","TAN","Sm","Sh","all_NIG","all_COTKEN","all_KENTAN","all_TANCOT"]
.index(trainset)]

def hotspot_conversion(y_true,y_pred,target):
	if target=="Prevalence Outcome":
		true_hotspots = y_true>.1
		pred_hotspots = y_pred>.1
	if target=="Intensity Outcome":
		true_hotspots = y_true>.01
		pred_hotspots = y_pred>.01
	if target=="Relative Outcome":
		true_hotspots = y_true<.35
		pred_hotspots = y_pred<.35
	return(true_hotspots,pred_hotspots)

if model in ['LinearRegression','LogisticRegression']:
	hotspot = ['Outcome','Hotspot']\
	[['LinearRegression','LogisticRegression'].index(model)]
	target = outcome+' '+hotspot
	filename = folder+var+val+'Trained_Models/'+model+'_'+trainset+'_'+outcome+'_'+hotspot+'_fit.pickle'
	csvname = folder+var+val+'Trained_Models/'+model+'_'+trainset+'_'+outcome+'_'+hotspot+'_variables.csv'
	full_variables = variable_func(trainset)
	with open(csvname,'r') as csvfile:
		reader = csv.reader(csvfile)
		select_variables = next(reader)
	with open(filename,'rb') as pklfile:
		reg_fit = pkl.load(pklfile)
	data = pd.read_csv(folder+"Test_Data/"+testset+".csv")
	X_unscaled = data[full_variables]
	y_true = np.array(data[target])
	with open(folder+"Train_Data/"+trainset+"_"+var+"scaler.pickle",'rb') as pklfile:
		scaler = pkl.load(pklfile)
	X_full = scaler.transform(X_unscaled)
	vindices =  [full_variables.index(variable) for variable in select_variables]
	X = np.array([X_full[:,v] for v in vindices]).T
else:
	hotspot = ["Outcome","Outcome","Outcome","Outcome","Hotspot","Hotspot","Hotspot","Hotspot"]\
	[['ElasticNetLinear','GradientBoostingRegressor','RandomForestRegressor','EnsembleRegressor'
	,'RandomForestClassifier','GradientBoostingClassifier','ElasticNetLogistic','EnsembleClassifier']
	.index(model)]
	target = outcome+' '+hotspot
	filename = folder+var+val+'Trained_Models/'+model+'_'+trainset+'_'+outcome+'_'+hotspot+'_fit.pickle'
	variables = variable_func(trainset)
	with open(filename,'rb') as pklfile:
		reg_fit = pkl.load(pklfile)
	data = pd.read_csv(folder+"Test_Data/"+testset+".csv")
	X_unscaled = data[variables]
	y_true = np.array(data[target])
	with open(folder+"Train_Data/"+trainset+"_"+var+"scaler.pickle",'rb') as pklfile:
		scaler = pkl.load(pklfile)
	X = scaler.transform(X_unscaled)

y_pred = reg_fit.predict(X)
if hotspot=='Outcome':
	y_true,y_pred = hotspot_conversion(y_true,y_pred,target)
if metric == 'Accuracy':
	p = np.mean(y_true==y_pred)
	CI = interval(p,len(y_pred),p_value)
elif metric == 'Balanced':
	sensitivity = np.mean([y_true[i]==y_pred[i] for i in range(len(y_true)) if y_true[i]])
	specificity = np.mean([y_true[i]==y_pred[i] for i in range(len(y_true)) if not y_true[i]])
	p =.5*(sensitivity+specificity)
	CI = interval(p,len(y_pred),p_value)
elif metric == 'Sensitivity':
	p=np.mean([y_true[i]==y_pred[i] for i in range(len(y_true)) if y_true[i]])
	CI = interval(p,sum(y_true),p_value)
elif metric == 'Specificity':
	p=np.mean([y_true[i]==y_pred[i] for i in range(len(y_true)) if not y_true[i]])
	CI = interval(p,sum(1-y_true),p_value)
elif metric == 'PPV':
	p=np.mean([y_true[i]==y_pred[i] for i in range(len(y_true)) if y_pred[i]])
	CI = interval(p,sum(y_pred),p_value)
elif metric == 'NPV':
	p=np.mean([y_true[i]==y_pred[i] for i in range(len(y_true)) if not y_pred[i]])
	CI = interval(p,1-sum(y_pred),p_value)
print('"'+f'{p*100:.0f}'+' ('+f'{CI[0]*100:.0f}'+'--'+f'{CI[1]*100:.0f}'+')"')
