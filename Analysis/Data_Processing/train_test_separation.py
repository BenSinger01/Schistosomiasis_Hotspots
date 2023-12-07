import pandas as pd
import numpy as np
import sys

folder, seed, validation_approach = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# folder = ''
# seed = 220516

if validation_approach=="Fixed":
	train_prop = 0.8
	val_prop = 0.2
	test_prop = 0.2
else:
	train_prop = 0.7
	val_prop = 0
	test_prop = 0.3

all_data = pd.read_csv("Data/all_hotspot_prediction_data.csv",index_col=[0])

np.random.seed(seed)

for country in ["NIG","MOZ","COT","KEN","TAN"]:
	n = all_data[all_data["Country_Code"]==country].shape[0]
	test_choice = np.random.choice(range(n), size=int(n*test_prop), replace=False)    
	test_ind = np.zeros(n, dtype=bool)
	test_ind[test_choice] = True
	train_ind = ~test_ind

	train = all_data[all_data["Country_Code"]==country].iloc[train_ind]
	test = all_data[all_data["Country_Code"]==country].iloc[test_ind]

	train.to_csv(folder+"Data/"+validation_approach+"Train_Data/"+country+".csv")
	test.to_csv(folder+"Data/"+validation_approach+"Test_Data/"+country+".csv")
	if validation_approach=="Fixed":
		n_train = np.sum(train_ind)
		val_choice = np.random.choice(range(n_train), size=int(n_train*val_prop/train_prop), replace=False)
		fold = -np.ones(n_train,dtype=int)
		fold[val_choice] = 0
		np.savetxt(folder+"Data/"+"FixedTrain_Data/"+country+"_fold.csv",fold,fmt='%1i')


train = all_data[all_data["Country_Code"]=="NIG"]
train.to_csv(folder+"Data/"+validation_approach+"Train_Data/all_NIG.csv")
test = all_data[all_data["Country_Code"]=="MOZ"]
test.to_csv(folder+"Data/"+validation_approach+"Test_Data/all_MOZ.csv")
if validation_approach=="Fixed":
	n_train = train.shape[0]
	val_choice = np.random.choice(range(n_train), size=int(n_train*val_prop/train_prop), replace=False)
	fold = -np.ones(n_train,dtype=int)
	fold[val_choice] = 0
	np.savetxt(folder+"Data/"+"FixedTrain_Data/all_NIG_fold.csv",fold,fmt='%1i')

for country1,country2 in [("COT","KEN"),("KEN","TAN"),("TAN","COT")]:
	train = all_data[(all_data["Country_Code"]==country1)|(all_data["Country_Code"]==country2)]
	train.to_csv(folder+"Data/"+validation_approach+"Train_Data/all_"+country1+country2+".csv")
	test = all_data[(all_data["Country_Code"]==country1)]
	test.to_csv(folder+"Data/"+validation_approach+"Test_Data/all_"+country1+".csv")
	if validation_approach=="Fixed":
		n_train = train.shape[0]
		val_choice = np.random.choice(range(n_train), size=int(n_train*val_prop/train_prop), replace=False)
		fold = -np.ones(n_train,dtype=int)
		fold[val_choice] = 0
		np.savetxt(folder+"Data/"+"FixedTrain_Data/all_"+country1+country2+"_fold.csv",fold,fmt='%1i')


for species in ["Sh","Sm"]:
	n = all_data[all_data["Species"]==species].shape[0]
	test_choice = np.random.choice(range(n), size=int(n*test_prop), replace=False)    
	test_ind = np.zeros(n, dtype=bool)
	test_ind[test_choice] = True
	train_ind = ~test_ind

	train = all_data[all_data["Species"]==species].iloc[train_ind]
	test = all_data[all_data["Species"]==species].iloc[test_ind]

	# all_data[all_data["Species"]==species].to_csv("Country_Data/"+species+".csv")
	train.to_csv(folder+"Data/"+validation_approach+"Train_Data/"+species+".csv")
	test.to_csv(folder+"Data/"+validation_approach+"Test_Data/"+species+".csv")
	if validation_approach=="Fixed":
		n_train = np.sum(train_ind)
		val_choice = np.random.choice(range(n_train), size=int(n_train*val_prop/train_prop), replace=False)
		fold = -np.ones(n_train,dtype=int)
		fold[val_choice] = 0
		np.savetxt(folder+"Data/"+"FixedTrain_Data/"+species+"_fold.csv",fold,fmt='%1i')
