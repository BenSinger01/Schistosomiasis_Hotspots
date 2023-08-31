import pandas as pd
import numpy as np
import sys
import wilson_score_interval

filename1, filename2 = sys.argv[1], sys.argv[2]

all_scores1 = pd.read_csv(filename1)
all_scores2 = pd.read_csv(filename2)

print("\\documentclass{standalone}")
print("\\usepackage{booktabs}")
print("\\begin{document}")
print("\\begin{tabular}{llrrrrrr}")
print("% \\multicolumn{11}{c}{\\textbf{Model accuracy (\\%)}}\\\\")
print("\\toprule")
print("& & \\multicolumn{3}{l}{\\textit{S. haematobium}} & \\multicolumn{3}{l}{\\textit{S. mansoni}}\\\\")
print("& & Accuracy & Sensitivity & Specificity & Accuracy & Sensitivity & Specificity\\\\")
df = pd.DataFrame()
for outcome in ['Prevalence','Prevalence Intensity','Prevalence Relative']:
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
	df_all_Sm1 = all_scores1[(all_scores1['Outcome']==outcome)&(all_scores1['Trainset']=="Sm")&(all_scores1['Model']!="EnsembleClassifier")&(all_scores1['Model']!="EnsembleRegressor")]\
    .pivot_table(values='Score',columns='Metric',index='Model'
        ).reindex(models)
	df_all_Sh1 = all_scores1[(all_scores1['Outcome']==outcome)&(all_scores1['Trainset']=="Sh")&(all_scores1['Model']!="EnsembleClassifier")&(all_scores1['Model']!="EnsembleRegressor")]\
    .pivot_table(values='Score',columns='Metric',index='Model'
        ).reindex(models)
	df_ens_Sm1 = all_scores1[(all_scores1['Outcome']==outcome)
        &(all_scores1['Model']=="EnsembleClassifier")
        &(all_scores1['Trainset']=="Sm")].pivot_table(
		values='Score',columns='Metric',index='Model').iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_ens_Sh1 = all_scores1[(all_scores1['Outcome']==outcome)
        &(all_scores1['Model']=="EnsembleClassifier")
        &(all_scores1['Trainset']=="Sh")].pivot_table(
		values='Score',columns='Metric',index='Model').iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_all_Sm2 = all_scores2[(all_scores2['Outcome']==outcome)&(all_scores2['Trainset']=="Sm")&(all_scores2['Model']!="EnsembleClassifier")&(all_scores2['Model']!="EnsembleRegressor")]\
    .pivot_table(values='Score',columns='Metric',index='Model'
        ).reindex(models)
	df_all_Sh2 = all_scores2[(all_scores2['Outcome']==outcome)&(all_scores2['Trainset']=="Sh")&(all_scores2['Model']!="EnsembleClassifier")&(all_scores2['Model']!="EnsembleRegressor")]\
    .pivot_table(values='Score',columns='Metric',index='Model'
        ).reindex(models)
	df_ens_Sm2 = all_scores2[(all_scores2['Outcome']==outcome)
        &(all_scores2['Model']=="EnsembleClassifier")
        &(all_scores2['Trainset']=="Sm")].pivot_table(
		values='Score',columns='Metric',index='Model').iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_ens_Sh2 = all_scores2[(all_scores2['Outcome']==outcome)
        &(all_scores2['Model']=="EnsembleClassifier")
        &(all_scores2['Trainset']=="Sh")].pivot_table(
		values='Score',columns='Metric',index='Model').iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_best_Sm1 = df_all_Sm1[df_all_Sm1['Accuracy']==df_all_Sm1['Accuracy'].max()].iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_best_Sh1 = df_all_Sh1[df_all_Sh1['Accuracy']==df_all_Sh1['Accuracy'].max()].iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_best_Sm2 = df_all_Sm2[df_all_Sm1['Accuracy']==df_all_Sm1['Accuracy'].max()].iloc[0][['Accuracy','Sensitivity','Specificity']]
	df_best_Sh2 = df_all_Sh2[df_all_Sh1['Accuracy']==df_all_Sh1['Accuracy'].max()].iloc[0][['Accuracy','Sensitivity','Specificity']]
	print("\\midrule")
	print(outcome_name+'&Best model',end="")
	for species in ['Sh','Sm']:
		for metric in ['Accuracy','Sensitivity','Specificity']:
			value = eval('df_best_'+species+'1')[metric] - eval('df_best_'+species+'2')[metric]
			print('&%.2f'%(value*100),end="")
	print('\\\\')
	print('hotspot&Ensemble',end="")
	for species in ['Sh','Sm']:
		for metric in ['Accuracy','Sensitivity','Specificity']:
			value = eval('df_ens_'+species+'1')[metric] - eval('df_ens_'+species+'2')[metric]
			print('&%.2f'%(value*100),end="")
	print('\\\\')
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{document}")