import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
import pickle as pkl
import csv
import sys
sys.path.append('Analysis')
import matplotlib.pyplot as plt
import variable_func
import validation_score

np.random.seed(230803)

# load control variables from command line
folder, var, validation_approach, plot = sys.argv[1], sys.argv[2], sys.argv[3], bool(int(sys.argv[4]))
reg_targets=["Prevalence Outcome"]
if var == "true_":
	reg_targets = ["True Prevalence Outcome","Prevalence Outcome"]
variable_func = eval('variable_func.'+var+'variable_func')
validation_score = validation_score.validation_score

parsimony_bias = 0.01

# select, train, and save linear regression models
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

	# select model for each hotspot definition
	for target in reg_targets:

		# load and initialize variables
		y = np.array(data[target])
		scorer = metrics.make_scorer(validation_score,target=target)
		reg = linear_model.LinearRegression()
		n_v = len(variables)
		variable_pool = np.arange(n_v)
		variable_selections = [[],]
		selection_scores = np.zeros(n_v+1)
		selection_se = np.zeros(n_v+1)

		# forward stepwise variable selection with cross-validation
		for step in range(n_v):
			scores = np.zeros(n_v-step)
			for variable in variable_pool:
				model_variables = variable_selections[-1].copy()
				model_variables.append(variable)
				X_subset = np.array([X[:,v] for v in model_variables]).T

				# groupwise cross-validation if between-country Sh, otherwise 5-fold
				if validation_approach=="Fixed":
					fold = np.loadtxt(folder+"Data/"+"FixedTrain_Data/"+trainset+"_fold.csv",delimiter=',')
					ps = model_selection.PredefinedSplit(test_fold=fold)
					scores[np.where(variable_pool==variable)[0]] = np.mean(
						model_selection.cross_val_score(reg,X_subset,y
							,cv=ps,scoring=scorer)
						)
				elif len(trainset)>7:
					scores[np.where(variable_pool==variable)[0]] = np.mean(
						model_selection.cross_val_score(reg,X_subset,y
							,cv=model_selection.LeaveOneGroupOut()
							,groups=country_indicator,scoring=scorer)
						)
				else:
					scores[np.where(variable_pool==variable)[0]] = np.mean(
						model_selection.cross_val_score(reg,X_subset,y
							,cv=5,scoring=scorer)
						)

			# store information about variables used
			variable_add = variable_pool[np.argmax(scores)]
			variable_pool = np.delete(variable_pool,np.argmax(scores))
			selection_scores[step+1]=np.max(scores)
			selection_se[step+1]=np.std(scores)/np.sqrt(len(scores))
			best_set = variable_selections[-1].copy()
			best_set.append(variable_add)
			variable_selections.append(best_set)

		# selection_scores[0] = np.maximum(validation_score(y,np.zeros(len(y)),target)
		# 	,validation_score(y,np.ones(len(y )),target))
		best_selection_score = selection_scores.max()
		if plot:
			# Create diagnostic plots
			fig, ax = plt.subplots()
			best_index = selection_scores.argmax()
			parsimony_biases = [0.01,0.02,selection_se[best_index]]
			ax.plot(1-selection_scores)
			for j in range(3):
				pb = parsimony_biases[j]
				color = ['#648FFF','#DC267F','#FFB000'][j]
				label = ['1%','2%','SE'][j]
				offset = [-.07,.07,0][j]
				best_biased_index = next(i for i,v in enumerate(selection_scores) 
					if v >= best_selection_score-pb)
				ax.axvline(best_biased_index+offset,c=color,label=label)
			ax.axvline(best_index,c='k',ls='--',label='best')
			ax.set_ylabel(target[:-8])
			ax.set_ylim([0,.5])
			ax.legend()
			ax.set_title(trainset+' linear regression')
			ax.set_xlabel('# variables')
			ax.set_xticks(range(0,n_v,2))
			plt.savefig(folder+"Outputs/"+var+validation_approach+"Trained_Models/LinearRegression_"
					+trainset+"_"+target.replace(' ','_')+"_figure.png")
			plt.close()

		# choose parsimony biased model
		best_biased_index = next(i for i,v in enumerate(selection_scores) 
			if v >= best_selection_score-parsimony_bias)
		best_biased_subset = variable_selections[best_biased_index]
		X_best_biased_subset = np.array([X[:,v] for v in best_biased_subset]).T
		reg_fit = reg.fit(X_best_biased_subset,y)
		y_pred = reg_fit.predict(X_best_biased_subset)
		best_biased_score = validation_score(y,y_pred,target)

		# edit regressor to take full set of variables,
		# so it can be included in the ensemble
		full_coefs = np.zeros(len(variables))
		for i in range(len(best_biased_subset)):
			full_coefs[best_biased_subset[i]] = reg_fit.coef_[i]
		reg_fit.coef_=full_coefs
		reg_fit.n_features_in_=len(full_coefs)

		# note coefficients and add intercept
		coefs = list(reg_fit.coef_)
		coefs.insert(0,reg_fit.intercept_)
		subset_names = [variables[i] for i in best_biased_subset]
		subset_names.insert(0,'Intercept')

		# save score
		with open(folder+"Outputs"+var+validation_approach+"Trained_Models/LinearRegression_"
			+trainset+"_"+target.replace(' ','_')+"_score.txt",'w') as wfile:
			wfile.write('%f' % best_biased_score)
		# save coefficients
		with open(folder+"Outputs"+var+validation_approach+"Trained_Models/LinearRegression_"
			+trainset+"_"+target.replace(' ','_')+".csv",'w') as wfile:
			writer = csv.writer(wfile)
			writer.writerow(subset_names)
			writer.writerow(coefs)
			writer.writerow([best_biased_score])
		# save pickled model
		with open(folder+"Outputs"+var+validation_approach+"Trained_Models/LinearRegression_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(reg_fit,wfile)
		# save list of variables used
		with open(folder+"Outputs"+var+validation_approach+"Trained_Models/LinearRegression_"
			+trainset+"_"+target.replace(' ','_')+"_variables.csv",'w') as wfile:
			writer = csv.writer(wfile)
			writer.writerow(subset_names[1:])