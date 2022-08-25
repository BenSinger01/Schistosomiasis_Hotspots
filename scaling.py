import pandas as pd
import pickle as pkl
from sklearn import preprocessing
import variable_func
import sys

folder, var = sys.argv[1], sys.argv[2]

variable_func = eval('variable_func.'+var+'variable_func')

for trainset in ["NIG","MOZ","COT","KEN","TAN","Sm","Sh"
,"all_NIG","all_COTKEN","all_KENTAN","all_TANCOT"]:
	data = pd.read_csv(folder+"Train_Data/"+trainset+".csv",index_col=[0])

	X = data[variable_func(trainset)]

	scaler = preprocessing.StandardScaler()
	fit_scaler = scaler.fit(X)
	with open(folder+"Train_Data/"+trainset+var+"_scaler.pickle",'wb') as pklfile:
		pkl.dump(fit_scaler,pklfile)

for trainset in ["NIG","MOZ","COT","KEN","TAN","Sm","Sh","all_NIG"]:
	data = pd.read_csv(folder+"Train_Data/"+trainset+".csv",index_col=[0])

	X = data[variable_func(trainset)]

	scaler = preprocessing.StandardScaler()
	fit_scaler = scaler.fit(X)
	X_scaled = scaler.transform(X)
	df_X_scaled = pd.DataFrame(X_scaled,columns=variable_func(trainset),index=X.index)
	ys = data[["Prevalence Outcome","Intensity Outcome","Relative Outcome"
	,"Prevalence Hotspot","Intensity Hotspot","Relative Hotspot"]]
	ys = ys.astype(int)

	df = df_X_scaled.join(ys)
	df.to_csv(folder+"Train_Data/"+trainset+var+"_scaled.csv")

for trainset in ["all_COTKEN","all_KENTAN","all_TANCOT"]:
	data = pd.read_csv(folder+"Train_Data/"+trainset+".csv",index_col=[0])

	X = data[variable_func(trainset)]

	scaler = preprocessing.StandardScaler()
	fit_scaler = scaler.fit(X)
	X_scaled = scaler.transform(X)
	df_X_scaled = pd.DataFrame(X_scaled,columns=variable_func(trainset),index=X.index)
	ys = data[["Prevalence Outcome","Intensity Outcome","Relative Outcome"
	,"Prevalence Hotspot","Intensity Hotspot","Relative Hotspot"]]
	ys = ys.astype(int)

	df = df_X_scaled.join(ys)
	df["Country.Code"] = data["Country_Code"]
	df.to_csv(folder+"Train_Data/"+trainset+var+"_scaled.csv")

