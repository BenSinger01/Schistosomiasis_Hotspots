from sklearn import metrics
from sklearn import model_selection
import pickle as pkl
import numpy as np
import pandas as pd
from wilson_score_interval import interval
import variable_func
import csv
import sys

model, trainset, outcome, p_value, folder, var, val, validation_approach =\
 sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]

variable_func = eval('variable_func.'+var+'variable_func')

testset = ["NIG","KEN","TAN","Sm","Sh","all_MOZ","all_TAN","all_COT","all_KEN"]\
[["NIG","KEN","TAN","Sm","Sh","all_NIG","all_COTKEN","all_KENTAN","all_TANCOT"]
.index(trainset)]

def hotspot_conversion(y_true,y_pred,target):
	if (target=="Prevalence Outcome")|(target=="True Prevalence Outcome"):
		true_hotspots = y_true>.1
		pred_hotspots = y_pred>.1
	if target=="Intensity Outcome":
		true_hotspots = y_true>.01
		pred_hotspots = y_pred>.01
	if target=="Relative Outcome":
		true_hotspots = y_true<.35
		pred_hotspots = y_pred<.35
	return(true_hotspots,pred_hotspots)

# if model in ['LinearRegression','LogisticRegression']:
# 	hotspot = ['Outcome','Hotspot']\
# 	[['LinearRegression','LogisticRegression'].index(model)]
# 	target = outcome+' '+hotspot
# 	filename = folder+var+val+validation_approach+'Trained_Models/'+model+'_'+trainset+'_'+outcome.replace(' ','_')+'_'+hotspot+'_fit.pickle'
# 	csvname = folder+var+val+validation_approach+'Trained_Models/'+model+'_'+trainset+'_'+outcome.replace(' ','_')+'_'+hotspot+'_variables.csv'
# 	full_variables = variable_func(trainset)
# 	with open(csvname,'r') as csvfile:
# 		reader = csv.reader(csvfile)
# 		select_variables = next(reader)
# 	with open(filename,'rb') as pklfile:
# 		reg_fit = pkl.load(pklfile)
# 	data = pd.read_csv(folder+validation_approach+"Test_Data/"+testset+".csv")
# 	X_unscaled = data[full_variables]
# 	y_true = np.array(data[target])
# 	with open(folder+validation_approach+"Train_Data/"+trainset+"_"+var+"scaler.pickle",'rb') as pklfile:
# 		scaler = pkl.load(pklfile)
# 	X_full = scaler.transform(X_unscaled)
# 	vindices =  [full_variables.index(variable) for variable in select_variables]
# 	X = np.array([X_full[:,v] for v in vindices]).T
# else:
hotspot = ["Outcome","Outcome","Outcome","Outcome","Outcome","Outcome","Outcome","Outcome","Outcome","Hotspot","Hotspot","Hotspot","Hotspot","Hotspot","Hotspot","Hotspot","Hotspot"]\
[['LinearRegression','ElasticNetLinear','GradientBoostingRegressor','XGBoostRegressor','RandomForestRegressor','SupportVectorRegressor','MultilayerPerceptronRegressor','EnsembleRegressor','SelectEnsembleRegressor'
,'LogisticRegression','ElasticNetLogistic','RandomForestClassifier','GradientBoostingClassifier','XGBoostClassifier','SupportVectorClassifier','MultilayerPerceptronClassifier','EnsembleClassifier','SelectEnsembleClassifier']
.index(model)]
target = outcome+' '+hotspot
filename = folder+var+val+validation_approach+'Trained_Models/'+model+'_'+trainset+'_'+outcome.replace(' ','_')+'_'+hotspot+'_fit.pickle'
variables = variable_func(trainset)
with open(filename,'rb') as pklfile:
	reg_fit = pkl.load(pklfile)
data = pd.read_csv(folder+validation_approach+"Test_Data/"+testset+".csv")
X_unscaled = data[variables]
y_true = np.array(data[target])
with open(folder+validation_approach+"Train_Data/"+trainset+"_"+var+"scaler.pickle",'rb') as pklfile:
	scaler = pkl.load(pklfile)
X = scaler.transform(X_unscaled)

if (hotspot == 'Hotspot') & (model != 'EnsembleClassifier'):
	try:
		y_score = reg_fit.predict_proba(X)[:,1]
	except:
		print(model+','+trainset+','+outcome+',AUC,nan')
	try:
		print(model+','+trainset+','+outcome+',AUC,'+str(metrics.roc_auc_score(y_true,y_score))+',nan,nan')
	except:
		print(model+','+trainset+','+outcome+',AUC,nan,nan,nan')
y_pred = reg_fit.predict(X)
if hotspot=='Outcome':
	y_true,y_pred = hotspot_conversion(y_true,y_pred,target)
accuracy = np.mean(y_true==y_pred)
CI_accuracy = interval(accuracy,len(y_pred),p_value)
print(model+','+trainset+','+outcome+',Accuracy,'+str(accuracy),','+str(CI_accuracy[0])+','+str(CI_accuracy[1]))
sensitivity = np.mean([y_true[i]==y_pred[i] for i in range(len(y_true)) if y_true[i]])
CI_sensitivity = interval(sensitivity,sum(y_true),p_value)
print(model+','+trainset+','+outcome+',Sensitivity,'+str(sensitivity),','+str(CI_sensitivity[0])+','+str(CI_sensitivity[1]))
specificity = np.mean([y_true[i]==y_pred[i] for i in range(len(y_true)) if not y_true[i]])
CI_specificity = interval(specificity,sum(1-y_true),p_value)
print(model+','+trainset+','+outcome+',Specificity,'+str(specificity),','+str(CI_specificity[0])+','+str(CI_specificity[1]))
fpr = np.mean([y_true[i]!=y_pred[i] for i in range(len(y_true)) if not y_true[i]])
CI_fpr = interval(fpr,sum(1-y_true),p_value)
print(model+','+trainset+','+outcome+',FPR,'+str(fpr)+','+str(CI_fpr[0])+','+str(CI_fpr[1]))
balanced = .5*(sensitivity+specificity)
CI_balanced = interval(balanced,len(y_pred),p_value)
print(model+','+trainset+','+outcome+',Balanced,'+str(balanced)+','+str(CI_balanced[0])+','+str(CI_balanced[1]))
ppv = np.mean([y_true[i]==y_pred[i] for i in range(len(y_true)) if y_pred[i]])
CI_ppv = interval(ppv,sum(y_pred),p_value)
print(model+','+trainset+','+outcome+',PPV,'+str(ppv)+','+str(CI_ppv[0])+','+str(CI_ppv[1]))
npv = np.mean([y_true[i]==y_pred[i] for i in range(len(y_true)) if not y_pred[i]])
CI_npv = interval(npv,sum(1-y_pred),p_value)
print(model+','+trainset+','+outcome+',NPV,'+str(npv)+','+str(CI_npv[0])+','+str(CI_npv[1]))
print(model+','+trainset+','+outcome+',F1,'+str(2*(sensitivity*ppv)/(sensitivity+ppv))+',nan,nan')
