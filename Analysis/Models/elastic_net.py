import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
import pickle as pkl
import csv
import sys
from matplotlib import pyplot as plt
import variable_func
import validation_score

np.random.seed(220517)

# load control variables from command line
folder, var, val, validation_approach, plot = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], bool(int(sys.argv[5]))
clas_targets=["Prevalence Hotspot","Prevalence Relative Hotspot","Prevalence Intensity Hotspot"]
reg_targets=["Prevalence Outcome"]
if var == "true_":
	clas_targets=["True Prevalence Hotspot","True Prevalence Relative Hotspot","True Prevalence Intensity Hotspot","Prevalence Hotspot","Prevalence Relative Hotspot","Prevalence Intensity Hotspot"]
	reg_targets = ["True Prevalence Outcome","Prevalence Outcome"]

variable_func = eval('variable_func.'+var+'variable_func')
validation_score = eval('validation_score.'+val+'validation_score')

# define hyperparamter search
# l1 ratios search more densely in upper range, as suggested in sklearn documentation
# alpha searches across several orders of magnitude
regressor_search_params = {
"alpha" : np.exp(np.linspace(-10,10,10))
,"l1_ratio" : np.log(np.linspace(1,np.e,10))
}
classifier_search_params = {
"C" : 1/np.exp(np.linspace(-10,10,10))
,"l1_ratio" : np.log(np.linspace(1,np.e,10))
}

# select, train, and save elastic net regularized GLMs
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
		reg = linear_model.ElasticNet(max_iter=int(1e6))

		# search hyperparameter space using groupwise cross-validation
		# if between-country Sh, otherwise 5-fold
		if validation_approach=="Fixed":
			fold = np.loadtxt(folder+"Data/"+"FixedTrain_Data/"+trainset+"_fold.csv",delimiter=',',dtype=int)
			ps = model_selection.PredefinedSplit(test_fold=fold)
			search = model_selection.GridSearchCV(reg,param_grid=regressor_search_params
				,scoring=scorer
				,cv=ps)
			search_fit = search.fit(X,y)
		elif len(trainset)>7:
			search = model_selection.GridSearchCV(reg,param_grid=regressor_search_params
				,scoring=scorer
				,cv=model_selection.LeaveOneGroupOut())
			search_fit = search.fit(X,y
				,groups=country_indicator)
		else:
			search = model_selection.GridSearchCV(reg,param_grid=regressor_search_params
				,scoring=scorer
				,cv=5)
			search_fit = search.fit(X,y)
		reg_fit = search_fit.best_estimator_
		reg_score = search_fit.best_score_

		if plot:
			# get cv_results_ and plot a heatmap of scores for each parameter combination
			val_results = pd.DataFrame(search_fit.cv_results_)
			score_array = val_results.pivot_table(index="param_alpha",columns="param_l1_ratio",values="mean_test_score")
			fig,ax=plt.subplots()
			im=ax.pcolor(score_array)
			fig.colorbar(im)
			# put the major ticks at the middle of each cell, and round tick label to 2 signficant figures
			ax.set_yticks(np.arange(0.5, len(score_array.index), 1), [round(item,2) for item in score_array.index])
			ax.set_xticks(np.arange(0.5, len(score_array.columns), 1), [round(item,2) for item in score_array.columns])
			# label axes
			ax.set_ylabel(list(regressor_search_params.keys())[0])
			ax.set_xlabel(list(regressor_search_params.keys())[1])
			plt.savefig(folder+"Outputs/"+var+val+validation_approach+"Trained_Models/ElasticNetLinear_"
				+trainset+"_"+target.replace(' ','_')+"_CV_Heatmap.png")
			plt.close()

		# save score
		with open(folder+"Outputs/"+var+val+validation_approach+"Trained_Models/ElasticNetLinear_"
			+trainset+"_"+target.replace(' ','_')+"_score.txt",'w') as wfile:
			wfile.write('%f' % reg_score)
		# save coefficients
		coefs = list(reg_fit.coef_)
		coefs.insert(0,reg_fit.intercept_)
		with open(folder+"Outputs/"+var+val+validation_approach+"Trained_Models/ElasticNetLinear_"
			+trainset+"_"+target.replace(' ','_')+".csv",'w') as wfile:
			writer = csv.writer(wfile)
			writer.writerow(coefs)
			writer.writerow(search_fit.best_params_.values())
		# save pickled model
		with open(folder+"Outputs/"+var+val+validation_approach+"Trained_Models/ElasticNetLinear_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(reg_fit,wfile)
		# save list of variables used
		with open(folder+"Outputs/"+var+val+validation_approach+"Trained_Models/ElasticNetLinear_"
			+trainset+"_"+target.replace(' ','_')+"_search.pickle",'wb') as wfile:
			pkl.dump(search_fit,wfile)

	# select classification model for each hotspot definition
	for target in clas_targets:

		# load and initialize variables
		y = np.array(data[target])
		reg = linear_model.LogisticRegression(
			penalty='elasticnet',solver='saga',max_iter=int(1e6))

		# search hyperparameter space using groupwise cross-validation
		# if between-country Sh, otherwise 5-fold
		if validation_approach=="Fixed":
			fold = np.loadtxt(folder+"Data/"+"FixedTrain_Data/"+trainset+"_fold.csv",delimiter=',')
			ps = model_selection.PredefinedSplit(test_fold=fold)
			search = model_selection.GridSearchCV(reg,param_grid=classifier_search_params
				,scoring=val+"accuracy"
				,cv=ps)
			search_fit = search.fit(X,y)
		elif len(trainset)>7:
			search = model_selection.GridSearchCV(reg
				,param_grid=classifier_search_params
				,scoring=val+"accuracy"
				,cv=model_selection.LeaveOneGroupOut())
			search_fit = search.fit(X,y
				,groups=country_indicator)
		else:
			search = model_selection.GridSearchCV(reg
				,param_grid=classifier_search_params
				,scoring=val+"accuracy"
				,cv=5)
			search_fit = search.fit(X,y)
		reg_fit = search_fit.best_estimator_
		reg_score = search_fit.best_score_

		if plot:
			# get cv_results_ and plot a heatmap of scores for each parameter combination
			val_results = pd.DataFrame(search_fit.cv_results_)
			score_array = val_results.pivot_table(index="param_C",columns="param_l1_ratio",values="mean_test_score")
			fig,ax=plt.subplots()
			im=ax.pcolor(score_array)
			fig.colorbar(im)
			# put the major ticks at the middle of each cell, and round tick label to 2 signficant figures
			ax.set_yticks(np.arange(0.5, len(score_array.index), 1), [round(item,2) for item in score_array.index])
			ax.set_xticks(np.arange(0.5, len(score_array.columns), 1), [round(item,2) for item in score_array.columns])
			# label axes
			ax.set_ylabel(list(classifier_search_params.keys())[0])
			ax.set_xlabel(list(classifier_search_params.keys())[1])
			plt.savefig(folder+"Outputs"+var+val+validation_approach+"Trained_Models/ElasticNetLogistic_"
				+trainset+"_"+target.replace(' ','_')+"_CV_Heatmap.png")
			plt.close()

		# save score
		with open(folder+"Outputs"+var+val+validation_approach+"Trained_Models/ElasticNetLogistic_"
			+trainset+"_"+target.replace(' ','_')+"_score.txt",'w') as wfile:
			wfile.write('%f' % reg_score)
		# save coefficients
		coefs = list(reg_fit.coef_[0])
		coefs.insert(0,reg_fit.intercept_[0])
		with open(folder+"Outputs"+var+val+validation_approach+"Trained_Models/ElasticNetLogistic_"
			+trainset+"_"+target.replace(' ','_')+".csv",'w') as wfile:
			writer = csv.writer(wfile)
			writer.writerow(coefs)
			writer.writerow(search_fit.best_params_.values())
		# save pickled model
		with open(folder+"Outputs"+var+val+validation_approach+"Trained_Models/ElasticNetLogistic_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(reg_fit,wfile)
		# save list of variables used
		with open(folder+"Outputs"+var+val+validation_approach+"Trained_Models/ElasticNetLogistic_"
			+trainset+"_"+target.replace(' ','_')+"_search.pickle",'wb') as wfile:
			pkl.dump(search_fit,wfile)
