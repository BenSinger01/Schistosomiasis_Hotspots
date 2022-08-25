import pandas as pd
import numpy as np
import sys

folder, seed = sys.argv[1], int(sys.argv[2])

all_data = pd.read_csv("Data/all_hotspot_prediction_data.csv",index_col=[0])
# np.random.seed(220516)

np.random.seed(seed)

for country in ["NIG","MOZ","COT","KEN","TAN"]:
	n = all_data[all_data["Country_Code"]==country].shape[0]
	test_choice = np.random.choice(range(n), size=int(n*.3), replace=False)    
	test_ind = np.zeros(n, dtype=bool)
	test_ind[test_choice] = True
	train_ind = ~test_ind

	train = all_data[all_data["Country_Code"]==country].iloc[train_ind]
	test = all_data[all_data["Country_Code"]==country].iloc[test_ind]

	train.to_csv(folder+"Train_Data/"+country+".csv")
	test.to_csv(folder+"Test_Data/"+country+".csv")

all_data[all_data["Country_Code"]=="NIG"].to_csv(folder+"Train_Data/all_NIG.csv")
all_data[all_data["Country_Code"]=="MOZ"].to_csv(folder+"Test_Data/all_MOZ.csv")

for country1,country2 in [("COT","KEN"),("KEN","TAN"),("TAN","COT")]:
	all_data[
	(all_data["Country_Code"]==country1)|(all_data["Country_Code"]==country2)
	].to_csv(folder+"Train_Data/all_"+country1+country2+".csv")
	all_data[
	(all_data["Country_Code"]==country1)
	].to_csv(folder+"Test_Data/all_"+country1+".csv")


for species in ["Sh","Sm"]:
	n = all_data[all_data["Species"]==species].shape[0]
	test_choice = np.random.choice(range(n), size=int(n*.3), replace=False)    
	test_ind = np.zeros(n, dtype=bool)
	test_ind[test_choice] = True
	train_ind = ~test_ind

	train = all_data[all_data["Species"]==species].iloc[train_ind]
	test = all_data[all_data["Species"]==species].iloc[test_ind]

	# all_data[all_data["Species"]==species].to_csv("Country_Data/"+species+".csv")
	train.to_csv(folder+"Train_Data/"+species+".csv")
	test.to_csv(folder+"Test_Data/"+species+".csv")
