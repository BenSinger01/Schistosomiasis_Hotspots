import numpy as np
from sklearn import metrics

def validation_score(y_true,y_pred,target):
	if target=="Prevalence Outcome" or target=="True Prevalence Outcome":
		true_hotspots = y_true>.1
		pred_hotspots = y_pred>.1
	if target=="Intensity Outcome":
		true_hotspots = y_true>.01
		pred_hotspots = y_pred>.01
	if target=="Relative Outcome":
		true_hotspots = y_true<.35
		pred_hotspots = y_pred<.35
	return(np.sum(true_hotspots==pred_hotspots)/len(true_hotspots))

def balanced_validation_score(y_true,y_pred,target):
	if target=="Prevalence Outcome":
		true_hotspots = y_true>.1
		pred_hotspots = y_pred>.1
	if target=="Intensity Outcome":
		true_hotspots = y_true>.01
		pred_hotspots = y_pred>.01
	if target=="Relative Outcome":
		true_hotspots = y_true<.35
		pred_hotspots = y_pred<.35
	return(metrics.balanced_accuracy_score(true_hotspots,pred_hotspots))