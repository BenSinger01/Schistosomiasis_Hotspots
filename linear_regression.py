import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
import pickle as pkl
import csv
import sys
# import matplotlib.pyplot as plt
import variable_func
import validation_score

folder, var, val = sys.argv[1], sys.argv[2], sys.argv[3]

variable_func = eval('variable_func.'+var+'variable_func')
validation_score = eval('validation_score.'+val+'validation_score')

parsimony_bias = 0.01

for trainset in ["NIG","KEN","TAN","Sm","Sh","all_NIG"
,"all_COTKEN","all_KENTAN","all_TANCOT"]:
	# fig, ax = plt.subplots(3,1,sharex=True)
	i=0
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
				# groupwise cross-validation if cross-country Sh, otherwise 5-fold
				if len(trainset)>7:
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
		# # Figures
		# best_index = selection_scores.argmax()
		# parsimony_biases = [0.01,0.02,selection_se[best_index]]
		# ax[i].plot(1-selection_scores)
		# for j in range(3):
		# 	pb = parsimony_biases[j]
		# 	color = ['c','y','m'][j]
		# 	label = ['1%','2%','SE'][j]
		# 	offset = [-.07,.07,0][j]
		# 	best_biased_index = next(i for i,v in enumerate(selection_scores) 
		# 		if v >= best_selection_score-pb)
		# 	ax[i].axvline(best_biased_index+offset,c=color,label=label)
		# ax[i].axvline(best_index,c='k',ls='--',label='best')
		# ax[i].set_ylabel(target[:-8])
		# ax[i].set_ylim([0,.5])	
		# i+=1
		# # Save model
		best_biased_index = next(i for i,v in enumerate(selection_scores) 
			if v >= best_selection_score-parsimony_bias)
		if best_biased_index == n_v:
			best_biased_index = next(i for i,v in enumerate(selection_scores) 
			if v >= best_selection_score-parsimony_bias*2)
		best_biased_subset = variable_selections[best_biased_index]
		X_best_biased_subset = np.array([X[:,v] for v in best_biased_subset]).T
		reg_fit = reg.fit(X_best_biased_subset,y)
		y_pred = reg_fit.predict(X_best_biased_subset)
		best_biased_score = validation_score(y,y_pred,target)
		coefs = list(reg_fit.coef_)
		coefs.insert(0,reg_fit.intercept_)
		subset_names = [variables[i] for i in best_biased_subset]
		subset_names.insert(0,'Intercept')
		with open(folder+var+val+"Trained_Models/LinearRegression_"
			+trainset+"_"+target.replace(' ','_')+".csv",'w') as wfile:
			writer = csv.writer(wfile)
			writer.writerow(subset_names)
			writer.writerow(coefs)
			writer.writerow([best_biased_score])
		with open(folder+var+val+"Trained_Models/LinearRegression_"
			+trainset+"_"+target.replace(' ','_')+"_fit.pickle",'wb') as wfile:
			pkl.dump(reg_fit,wfile)
		with open(folder+var+val+"Trained_Models/LinearRegression_"
			+trainset+"_"+target.replace(' ','_')+"_variables.csv",'w') as wfile:
			writer = csv.writer(wfile)
			writer.writerow(subset_names[1:])
	# ax[0].set_title(trainset+' linear regression')
	# ax[2].set_xlabel('# variables')
	# ax[2].set_xticks(range(0,n_v,2))
	# ax[2].legend()
	# plt.savefig("Trained_Models/biased_LinearRegression_"
	# 		+trainset+"_figure.png")