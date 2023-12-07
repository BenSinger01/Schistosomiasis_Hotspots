import pandas as pd
import numpy as np
import sys
import wilson_score_interval

filename, CI, var = sys.argv[1], bool(sys.argv[2]), sys.argv[3]

all_scores = pd.read_csv(filename)
if var=='true_':
	outcome_list = ['True Prevalence','True Prevalence Intensity','True Prevalence Relative']
else:
	outcome_list = ['Prevalence','Prevalence Intensity','Prevalence Relative']

print("\\documentclass{standalone}")
print("\\usepackage{booktabs}")
print("\\begin{document}")
print("\\begin{tabular}{llrrrrrr}")
print("% \\multicolumn{11}{c}{\\textbf{Model accuracy (\\%)}}\\\\")
print("\\toprule")
print("& & \\multicolumn{3}{l}{\\textit{S. haematobium}} & \\multicolumn{3}{l}{\\textit{S. mansoni}}\\\\")
print("& & Accuracy & Sensitivity & Specificity & Accuracy & Sensitivity & Specificity\\\\")
df = pd.DataFrame()
for outcome in outcome_list:
	outcome_name = outcome.split(' ')[-1]
	models = [
	"LogisticRegression","ElasticNetLogistic",
	"RandomForestClassifier","GradientBoostingClassifier",
	"SupportVectorClassifier","MultilayerPerceptronClassifier"]
	if outcome == 'Prevalence':
		models = [
		"LogisticRegression","LinearRegression",
		"ElasticNetLogistic","ElasticNetLinear",
		"RandomForestClassifier","RandomForestRegressor",
		"GradientBoostingClassifier","GradientBoostingRegressor",
		"SupportVectorClassifier","SupportVectorRegressor",
		"MultilayerPerceptronClassifier","MultilayerPerceptronRegressor"
		]
	df_all_Sm = all_scores[(all_scores['Outcome']==outcome)&(all_scores['Trainset']=="Sm")&(all_scores['Model']!="EnsembleClassifier")&(all_scores['Model']!="EnsembleRegressor")]\
    .pivot_table(values='Score',columns='Metric',index='Model'
        ).reindex(models)
	df_all_Sm_lower = all_scores[(all_scores['Outcome']==outcome)&(all_scores['Trainset']=="Sm")&(all_scores['Model']!="EnsembleClassifier")&(all_scores['Model']!="EnsembleRegressor")]\
    .pivot_table(values='Lower',columns='Metric',index='Model'
        ).reindex(models)
	df_all_Sm_upper = all_scores[(all_scores['Outcome']==outcome)&(all_scores['Trainset']=="Sm")&(all_scores['Model']!="EnsembleClassifier")&(all_scores['Model']!="EnsembleRegressor")]\
    .pivot_table(values='Upper',columns='Metric',index='Model'
        ).reindex(models)
	df_all_Sh = all_scores[(all_scores['Outcome']==outcome)&(all_scores['Trainset']=="Sh")&(all_scores['Model']!="EnsembleClassifier")&(all_scores['Model']!="EnsembleRegressor")]\
    .pivot_table(values='Score',columns='Metric',index='Model'
        ).reindex(models)
	df_all_Sh_lower = all_scores[(all_scores['Outcome']==outcome)&(all_scores['Trainset']=="Sh")&(all_scores['Model']!="EnsembleClassifier")&(all_scores['Model']!="EnsembleRegressor")]\
	.pivot_table(values='Lower',columns='Metric',index='Model'
		).reindex(models)
	df_all_Sh_upper = all_scores[(all_scores['Outcome']==outcome)&(all_scores['Trainset']=="Sh")&(all_scores['Model']!="EnsembleClassifier")&(all_scores['Model']!="EnsembleRegressor")]\
	.pivot_table(values='Upper',columns='Metric',index='Model'
		).reindex(models)
	df_ens_Sm = all_scores[(all_scores['Outcome']==outcome)
        &(all_scores['Model']=="EnsembleClassifier")
        &(all_scores['Trainset']=="Sm")].pivot_table(
		values='Score',columns='Metric',index='Model').iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_ens_Sm_lower = all_scores[(all_scores['Outcome']==outcome)
		&(all_scores['Model']=="EnsembleClassifier")
		&(all_scores['Trainset']=="Sm")].pivot_table(
		values='Lower',columns='Metric',index='Model').iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_ens_Sm_upper = all_scores[(all_scores['Outcome']==outcome)
		&(all_scores['Model']=="EnsembleClassifier")
		&(all_scores['Trainset']=="Sm")].pivot_table(
		values='Upper',columns='Metric',index='Model').iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_ens_Sh = all_scores[(all_scores['Outcome']==outcome)
        &(all_scores['Model']=="EnsembleClassifier")
        &(all_scores['Trainset']=="Sh")].pivot_table(
		values='Score',columns='Metric',index='Model').iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_ens_Sh_lower = all_scores[(all_scores['Outcome']==outcome)
		&(all_scores['Model']=="EnsembleClassifier")
		&(all_scores['Trainset']=="Sh")].pivot_table(
		values='Lower',columns='Metric',index='Model').iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_ens_Sh_upper = all_scores[(all_scores['Outcome']==outcome)
		&(all_scores['Model']=="EnsembleClassifier")
		&(all_scores['Trainset']=="Sh")].pivot_table(
		values='Upper',columns='Metric',index='Model').iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_best_Sm = df_all_Sm[df_all_Sm['Accuracy']==df_all_Sm['Accuracy'].max()].iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_best_Sm_lower = df_all_Sm_lower[df_all_Sm['Accuracy']==df_all_Sm['Accuracy'].max()].iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_best_Sm_upper = df_all_Sm_upper[df_all_Sm['Accuracy']==df_all_Sm['Accuracy'].max()].iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_best_Sh = df_all_Sh[df_all_Sh['Accuracy']==df_all_Sh['Accuracy'].max()].iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_best_Sh_lower = df_all_Sh_lower[df_all_Sh['Accuracy']==df_all_Sh['Accuracy'].max()].iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_best_Sh_upper = df_all_Sh_upper[df_all_Sh['Accuracy']==df_all_Sh['Accuracy'].max()].iloc[0][['Accuracy','Sensitivity','Specificity']]
	print("\\midrule")
	print(outcome_name+'&Best model',end="")
	for species in ['Sh','Sm']:
		for metric in ['Accuracy','Sensitivity','Specificity']:
			value = eval('df_best_'+species)[metric]
			if CI:
				lower = eval('df_best_'+species+'_lower')[metric]
				upper = eval('df_best_'+species+'_upper')[metric]
				print('&%.0f (%.0f--%.0f)'%(value*100,lower*100,upper*100),end="")
			else:
				print('&%.0f'%(value*100),end="")
	print('\\\\')
	print('hotspot&Ensemble',end="")
	for species in ['Sh','Sm']:
		for metric in ['Accuracy','Sensitivity','Specificity']:
			value = eval('df_ens_'+species)[metric]
			if CI:
				lower = eval('df_ens_'+species+'_lower')[metric]
				upper = eval('df_ens_'+species+'_upper')[metric]
				print('&%.0f (%.0f--%.0f)'%(value*100,lower*100,upper*100),end="")
			else:
				print('&%.0f'%(value*100),end="")
	print('\\\\')
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{document}")