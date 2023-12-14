import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn import ensemble
import pickle as pkl
import csv
import sys
sys.path.append('Analysis')
import variable_func
import validation_score
from matplotlib import pyplot as plt

np.random.seed(220517)

# load control variables from command line
folder, var, validation_approach, plot = sys.argv[1], sys.argv[2], sys.argv[3], bool(int(sys.argv[4]))
las_targets=["Prevalence Hotspot","Prevalence Relative Hotspot","Prevalence Intensity Hotspot"]
clas_targets=["Prevalence Hotspot","Prevalence Relative Hotspot","Prevalence Intensity Hotspot"]
reg_targets=["Prevalence Outcome"]
if var == "true_":
	clas_targets=["True Prevalence Hotspot","True Prevalence Relative Hotspot","True Prevalence Intensity Hotspot","Prevalence Hotspot","Prevalence Relative Hotspot","Prevalence Intensity Hotspot"]
	reg_targets = ["True Prevalence Outcome","Prevalence Outcome"]

variable_func = eval('variable_func.'+var+'variable_func')
validation_score = validation_score.validation_score

# define hyperparameter search
search_params = {
"max_features" : np.linspace(1/26,1,10)
,"max_samples" : np.linspace(.1,.99,10)
}

# select, train, and save random forest models
# iterate over train sets (and corresponding test sets)
for trainset in ["NIG","KEN","TAN","Sm","Sh","all_NIG"
,"all_COTKEN","all_KENTAN","all_TANCOT"]:

	# load data and variables
	data = pd.read_csv(folder+"Data/"+validation_approach+"Train_Data/"+trainset+".csv")
	country_indicator =\
	 [item==data["Country_Code"].iloc[0] for item in data["Country_Code"]]
	variables = variable_func(trainset)
	X_unscaled = data[variables]

	# scale variables
	with open(folder+"Data/"+validation_approach+"Train_Data/"+trainset+"_"+var+"scaler.pickle",'rb') as pklfile:
		scaler = pkl.load(pklfile)
	X = scaler.transform(X_unscaled)

	# select regression model for each hotspot definition
	for target in reg_targets:

		# load and initialize variables
		y = np.array(data[target])
		scorer = metrics.make_scorer(validation_score,target=target)
		reg = ensemble.RandomForestRegressor(n_estimators=300)

		# search hyperparameter space using groupwise cross-validation
		# if between-country Sh, otherwise 5-fold
		if validation_approach=="Fixed":
			fold = np.loadtxt(folder+"Data/"+"FixedTrain_Data/"+trainset+"_fold.csv",delimiter=',',dtype=int)
			ps = model_selection.PredefinedSplit(test_fold=fold)
			search = model_selection.GridSearchCV(reg,param_grid=search_params
				,scoring=scorer
				,cv=ps)
			search_fit = search.fit(X,y)
		elif len(trainset)>7:
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
		reg_score = search_fit.best_score_

		if plot:
			# get cv_results_ and plot a heatmap of scores for each parameter combination
			val_results = pd.DataFrame(search_fit.cv_results_)
			score_array = val_results.pivot_table(index="param_max_features",columns="param_max_samples",values="mean_test_score")
			fig,ax=plt.subplots()
			im=ax.pcolor(score_array)
			fig.colorbar(im)
			# put the major ticks at the middle of each cell, and round tick label to 2 signficant figures
			ax.set_yticks(np.arange(0.5, len(score_array.index), 1), [round(item,2) for item in score_array.index])
			ax.set_xticks(np.arange(0.5, len(score_array.columns), 1), [round(item,2) for item in score_array.columns])
			# label axes
			ax.set_ylabel(list(search_params.keys())[0])
			ax.set_xlabel(list(search_params.keys())[1])
			plt.savefig(folder+"Outputs/"+var+validation_approach+"Trained_Models/RandomForestRegressor_"
				+trainset+"_"+target.replace(' ','_')+"_CV_Heatmap.png")
			plt.close()

		# save score
		with open(folder+"Outputs/"+var+validation_approach+"Trained_Models/RandomForestRegressor_"
			+trainset+"_"+target.replace(' ','_')+"_score.txt",'w') as wfile:
			wfile.write('%f' % reg_score)
		# save pickled model
		with open(folder+"Outputs/"+var+validation_approach+"Trained_Models/RandomForestRegressor_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(reg_fit,wfile)
		# save search
		with open(folder+"Outputs/"+var+validation_approach+"Trained_Models/RandomForestRegressor_"
			+trainset+"_"+target.replace(' ','_')+"_search.pickle",'wb') as wfile:
			pkl.dump(search_fit,wfile)

	# select classification model for each hotspot definition
	for target in clas_targets:

		# load and initialize variables
		y = np.array(data[target])
		clas = ensemble.RandomForestClassifier(n_estimators=300)

		# search hyperparameter space using groupwise cross-validation
		# if between-country Sh, otherwise 5-fold
		if validation_approach=="Fixed":
					fold = np.loadtxt(folder+"Data/"+"FixedTrain_Data/"+trainset+"_fold.csv",delimiter=',')
					ps = model_selection.PredefinedSplit(test_fold=fold)
					search = model_selection.GridSearchCV(clas,param_grid=search_params
						,scoring="accuracy"
						,cv=ps)
					search_fit = search.fit(X,y)
		elif len(trainset)>7:
			search = model_selection.GridSearchCV(clas,param_grid=search_params
				,scoring="accuracy"
				,cv=model_selection.LeaveOneGroupOut())
			search_fit = search.fit(X,y
				,groups=country_indicator)
		else:
			search = model_selection.GridSearchCV(clas,param_grid=search_params
				,scoring="accuracy"
				,cv=5)
			search_fit = search.fit(X,y)
		clas_fit = search_fit.best_estimator_
		clas_score = search_fit.best_score_

		if plot:
			# get cv_results_ and plot a heatmap of scores for each parameter combination
			val_results = pd.DataFrame(search_fit.cv_results_)
			score_array = val_results.pivot_table(index="param_max_features",columns="param_max_samples",values="mean_test_score")
			fig,ax=plt.subplots()
			im=ax.pcolor(score_array)
			fig.colorbar(im)
			# put the major ticks at the middle of each cell, and round tick label to 2 signficant figures
			ax.set_yticks(np.arange(0.5, len(score_array.index), 1), [round(item,2) for item in score_array.index])
			ax.set_xticks(np.arange(0.5, len(score_array.columns), 1), [round(item,2) for item in score_array.columns])
			# label axes
			ax.set_ylabel(list(search_params.keys())[0])
			ax.set_xlabel(list(search_params.keys())[1])
			plt.savefig(folder+"Outputs/"+var+validation_approach+"Trained_Models/RandomForestClassifier_"
				+trainset+"_"+target.replace(' ','_')+"_CV_Heatmap.png")
			plt.close()

		# save score
		with open(folder+"Outputs/"+var+validation_approach+"Trained_Models/RandomForestClassifier_"
			+trainset+"_"+target.replace(' ','_')+"_score.txt",'w') as wfile:
			wfile.write('%f' % clas_score)
		# save pickled model
		with open(folder+"Outputs/"+var+validation_approach+"Trained_Models/RandomForestClassifier_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(clas_fit,wfile)
		# save search
		with open(folder+"Outputs/"+var+validation_approach+"Trained_Models/RandomForestClassifier_"
			+trainset+"_"+target.replace(' ','_')+"_search.pickle",'wb') as wfile:
			pkl.dump(search_fit,wfile)