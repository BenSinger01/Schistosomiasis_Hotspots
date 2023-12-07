import pandas as pd
import pickle as pkl
from sklearn import preprocessing
import variable_func
import sys

# load control variables from command line
folder, var, validation_approach = sys.argv[1], sys.argv[2], sys.argv[3]

variable_func = eval('variable_func.'+var+'variable_func')

# create scalers
for trainset in ["NIG","MOZ","COT","KEN","TAN","Sm","Sh"
,"all_NIG","all_COTKEN","all_KENTAN","all_TANCOT"]:
	data = pd.read_csv(folder+validation_approach+"Train_Data/"+trainset+".csv",index_col=[0])

	X = data[variable_func(trainset)]

	scaler = preprocessing.StandardScaler()
	fit_scaler = scaler.fit(X)
	with open(folder+validation_approach+"Train_Data/"+trainset+"_"+var+"scaler.pickle",'wb') as pklfile:
		pkl.dump(fit_scaler,pklfile)
