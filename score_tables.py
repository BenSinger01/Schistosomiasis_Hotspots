import pandas as pd
import numpy as np
import sys
import wilson_score_interval

filename, metric, CI, var = sys.argv[1], sys.argv[2], bool(int(sys.argv[3])), sys.argv[4]

all_scores = pd.read_csv(filename)
all_scores = all_scores[all_scores['Metric']==metric]
trainsets = ['Sh','all_NIG','NIG','Sm','all_KENTAN','all_TANCOT','all_COTKEN','KEN','TAN']
if var=='true_':
	outcome_list = ['True Prevalence','True Prevalence Intensity','True Prevalence Relative']
else:
	outcome_list = ['Prevalence','Prevalence Intensity','Prevalence Relative']

print("\\documentclass{standalone}")
print("\\usepackage{booktabs}")
print("\\begin{document}")
print("\\begin{tabular}{llrrrrrrrrr}")
print("% \\multicolumn{11}{c}{\\textbf{Model "+metric+" (\\%)}}\\\\")
print("\\toprule")
print("\\multicolumn{2}{l}{Species} & \\multicolumn{3}{l}{\\textit{S. haematobium}} & \\multicolumn{6}{l}{\\textit{S. mansoni}}\\\\")
print("\\midrule")
if filename[0]=="F":
	print("& & Combined* & Between & Within & Combined* & \\multicolumn{3}{l}{Between} & \\multicolumn{2}{l}{Within}\\\\")
	print("\\multicolumn{2}{l}{Test data country} & & MOZ* & NER* & & CIV* & KEN & TZA & KEN* & TZA* \\\\")
else:
	print("& & Combined & Between & Within & Combined & \\multicolumn{3}{l}{Between} & \\multicolumn{2}{l}{Within}\\\\")
	print("\\multicolumn{2}{l}{Test data country} & & MOZ* & NER & & CIV* & KEN & TZA & KEN & TZA* \\\\")
df = pd.DataFrame()
for outcome in outcome_list:
	outcome_name = outcome.split(' ')[-1]
	models = ["EnsembleClassifier",
	"LogisticRegression","ElasticNetLogistic",
	"RandomForestClassifier","GradientBoostingClassifier",
	"SupportVectorClassifier","MultilayerPerceptronClassifier"]
	if ((outcome == 'Prevalence')|(outcome == 'True Prevalence'))&(not(metric=='AUC')):
		models = [
		"EnsembleClassifier",
		"LogisticRegression","ElasticNetLogistic",
		"RandomForestClassifier","GradientBoostingClassifier",
		"SupportVectorClassifier","MultilayerPerceptronClassifier",
		"EnsembleRegressor",
		"LinearRegression","ElasticNetLinear",
		"RandomForestRegressor","GradientBoostingRegressor",
		"SupportVectorRegressor","MultilayerPerceptronRegressor"
		]
	if CI:
		df_temp = all_scores[(all_scores['Outcome']==outcome)]\
		.pivot_table(values='Score',columns='Trainset',index='Model'
			,dropna=False).reindex(models)
		df_lower = all_scores[(all_scores['Outcome']==outcome)]\
		.pivot_table(values='Lower',columns='Trainset',index='Model'
			,dropna=False).reindex(models)
		df_upper = all_scores[(all_scores['Outcome']==outcome)]\
		.pivot_table(values='Upper',columns='Trainset',index='Model'
			,dropna=False).reindex(models)
	else:
		df_temp = all_scores[(all_scores['Outcome']==outcome)]\
		.pivot_table(values='Score',columns='Trainset',index='Model'
			,dropna=False).reindex(models)
	print("\\midrule")
	for model in models:
		#check that row for model is not all nan
		if df_temp.loc[df_temp.index==model,:].isnull().all(axis=1).values[0]:
			continue
		if model=="EnsembleClassifier":
			print(outcome_name,end="")
		elif model=="LogisticRegression":
			print("hotspot",end="")
		print('&'+model,end="")
		for trainset in trainsets:
			value = df_temp.loc[df_temp.index==model,trainset].iloc[0]
			if np.isnan(value):
				print('&---',end="")
			elif CI:
				lower = df_lower.loc[df_lower.index==model,trainset].iloc[0]
				upper = df_upper.loc[df_upper.index==model,trainset].iloc[0]
				print('&%.0f (%.0f--%.0f)'%(value*100,lower*100,upper*100),end="")
			else:
				print('&%.0f'%(value*100),end="")
		print('\\\\')
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{document}")