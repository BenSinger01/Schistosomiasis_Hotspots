import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import expit
from nbinom_fit import fit_nbinom

# df contains all final diata
df = pd.DataFrame()

# Load SCORE data sets, plus any re-formating, and exclude villages with 
# insufficient coverage

# Nigeria
scoreNIG_full = pd.read_csv('Data/rls0021_SCORE_Niger_Sh_XS_Y12345.csv'
	,dtype={'Mean_eggs':'float'})
scoreNIG_full["Percent_SAC_Rxed_Rx1"] = pd.to_numeric(
	scoreNIG_full['Percent_SAC_Rxed_Rx1'], errors='coerce')
scoreNIG_full["Percent_SAC_Rxed_Rx2"] = pd.to_numeric(
	scoreNIG_full['Percent_SAC_Rxed_Rx2'], errors='coerce')
coverageNIG1 = scoreNIG_full.pivot_table(index='Village_ID',columns='Study_Year'
	,values='Percent_SAC_Rxed_Rx1')
coverageNIG2 = scoreNIG_full.pivot_table(index='Village_ID',columns='Study_Year'
	,values='Percent_SAC_Rxed_Rx2')
coverageNIG = pd.concat((coverageNIG1,coverageNIG2),axis=1)
cNIGarray = np.array(coverageNIG)
suff_coverage_NIG = pd.Series(np.sum(cNIGarray>75,axis=1)>=2,index=coverageNIG.index)
scoreNIG = scoreNIG_full[[suff_coverage_NIG.loc[ID] 
for ID in scoreNIG_full['Village_ID']]]

# Mozambique
scoreMOZ_full = pd.read_csv('Data/SCORE_Sh_MOZ_remove_flag03_exclusions.csv'
	,dtype={'Mean_eggs':'float'})
scoreMOZ_full["Percent_SAC_Rxed"] = pd.to_numeric(
	scoreMOZ_full['Percent_SAC_Rxed'], errors='coerce')
coverageMOZ = scoreMOZ_full.pivot_table(index='Village_ID',columns='Study_Year'
	,values='Percent_SAC_Rxed')
cMOZarray = np.array(coverageMOZ)
suff_coverage_MOZ = pd.Series(np.sum(cMOZarray>75,axis=1)>=2,index=coverageMOZ.index)
scoreMOZ = scoreMOZ_full[[suff_coverage_MOZ.loc[ID] 
for ID in scoreMOZ_full['Village_ID']]]

# Mansoni countries
scoreSm_full = pd.read_csv('Data/rls0021_SCORE_Sm_Y12345_Aug15_2019_TAN_exclusions.csv'
	,dtype={'Mean_eggs':'float'})
scoreSm_full["Percent_SAC_Rxed"] = pd.to_numeric(
	scoreSm_full['Percent_SAC_Rxed'], errors='coerce')
coverageSm = scoreSm_full.pivot_table(
	index='Village_ID',columns='Study_Year',values='Percent_SAC_Rxed')
cSmarray = np.array(coverageSm)
suff_coverage_Sm = pd.Series(np.sum(cSmarray>75,axis=1)>=2,index=coverageSm.index)
scoreSm = scoreSm_full[[suff_coverage_Sm.loc[ID] 
for ID in scoreSm_full['Village_ID']]]


########################### Basic variables ###########################

# Set village ID order, and basic identifiers
df["Village_ID"] = np.concatenate(
	(scoreNIG['Village_ID'].unique()
	,scoreMOZ['Village_ID'].unique()
	,scoreSm['Village_ID'].unique()))
df["Country_Code"] = [ID[:3] for ID in df["Village_ID"]]
df["Dummy_MOZ"] = df["Country_Code"] == "MOZ"
df["Dummy_NIG"] = df["Country_Code"] == "NIG"
df["Dummy_COT"] = df["Country_Code"] == "COT"
df["Dummy_KEN"] = df["Country_Code"] == "KEN"
df["Dummy_TAN"] = df["Country_Code"] == "TAN"
df["Species"] = [["Sm","Sh"][(code=='NIG') or (code=='MOZ')] 
for code in df["Country_Code"]]

# Compute rounds-of-MDA variables
# Niger is weird because the strategy changed half way through
NIG_Rx_data = scoreNIG.pivot_table(index="Village_ID"
	,values=["Niger_Rx","Study_Type","Study_Arm_Original"]
	,aggfunc = lambda x: pd.unique(x)[0])
SAC_Rx_array_NIG = np.array([
	[[4,4,3,0,0,0],
	 [6,6,5,0,0,0]],
	[[4,4,4,4,4,3],
	 [6,6,6,6,6,5]]
	])
NIG_SAC_Rxs = [SAC_Rx_array_NIG[['Sustaining','Gaining'].index(row.Study_Type)
,[' once','twice'].index(row.Niger_Rx[-5:]),row.Study_Arm_Original-1]
 for row in NIG_Rx_data.itertuples()]
NIG_Comm_Rxs = [[0,0,4,6]
[['SBT once','SBT twice','CWT once','CWT twice'].index(row.Niger_Rx)]
 for row in NIG_Rx_data.itertuples()]
NIG_Rx = pd.DataFrame(
	data = {"SAC Rounds":NIG_SAC_Rxs,"Community Rounds":NIG_Comm_Rxs}
	,index=NIG_Rx_data.index)

# Mozambique only had Gaining Control type studies
MOZ_Rx_data = scoreMOZ.pivot_table(index="Village_ID"
	,values=["Study_Arm"]
	,aggfunc = lambda x: pd.unique(x)[0])
SAC_Rx_array_MOZ = np.array([4,4,2,4,2,2])
Comm_Rx_array_MOZ = np.array([4,2,2,0,0,0])
MOZ_SAC_Rxs = [SAC_Rx_array_MOZ[row.Study_Arm-1] 
for row in MOZ_Rx_data.itertuples()]
MOZ_Comm_Rxs = [Comm_Rx_array_MOZ[row.Study_Arm-1] 
for row in MOZ_Rx_data.itertuples()]
MOZ_Rx = pd.DataFrame(
	data = {"SAC Rounds":MOZ_SAC_Rxs,"Community Rounds":MOZ_Comm_Rxs}
	,index=MOZ_Rx_data.index)

# mansoni
Sm_Rx_data = scoreSm.pivot_table(index="Village_ID"
	,values=["Study_Type","Study_Arm"]
	,aggfunc = lambda x: pd.unique(x)[0])
SAC_Rx_array_Sm = np.array([
	[4,2,2,0,0,0],
	[4,4,2,4,2,2]
	])
Comm_Rx_array_Sm = np.array([
	[0,0,0,0,0,0],
	[4,2,2,0,0,0]
	])
Sm_SAC_Rxs = [SAC_Rx_array_Sm[['Sustaining','Gaining'].index(row.Study_Type)
,row.Study_Arm-1] for row in Sm_Rx_data.itertuples()]
Sm_Comm_Rxs = [Comm_Rx_array_Sm[['Sustaining','Gaining'].index(row.Study_Type)
,row.Study_Arm-1] for row in Sm_Rx_data.itertuples()]
Sm_Rx = pd.DataFrame(
	data = {"SAC Rounds":Sm_SAC_Rxs,"Community Rounds":Sm_Comm_Rxs}
	,index=Sm_Rx_data.index)

# merge MDA data into overall dataframe
df = df.merge(pd.concat((NIG_Rx,MOZ_Rx,Sm_Rx)),how='left',on='Village_ID')


########################### Egg variables ###########################
# Compute variables based on Mean_epg/Mean_Eggs
# First, define some useful functions:
def mean_func(x):
	if not np.any(x>0):
		return(0)
	else:
		return(np.mean([y for y in x if y>0]))
def log_mean_func(x):
	if not np.any(x>0):
		return(0)
	else:
		return(np.log(np.mean([y for y in x if y>0])+1))
def gmean_barenbold_func(x):
	if not np.any(x>0):
		return(0)
	else:
		return(np.exp(np.mean(np.log(x/24+1)))-1)
def g_interaction(x):
	if not np.any(x>0):
		return(0)
	else:
		return(np.log(stats.gmean([y for y in x if y>0])*np.mean(x>0)+1))
def a_interaction(x):
	if not np.any(x>0):
		return(0)
	else:
		return(np.log(np.mean([y for y in x if y>0])*np.mean(x>0)+1))
def NB_fit(x):
	fit= fit_nbinom(x)
	n = fit["size"]
	return(1/n)
def log_NB_func(x):
	if not np.any(x>0):
		return(0)
	else:
		fit = fit_nbinom(np.array([y for y in x if y>0]))
		n = fit["size"]
		return(np.log(1+1/n))

# Calculate for Niger
NIG_epg_y1_allages = scoreNIG[scoreNIG["Study_Year"]==1].pivot_table(
	values="Mean_eggs",index="Village_ID"
	,aggfunc = ( log_NB_func
		, lambda x : np.mean(x>0)
		, g_interaction
		, a_interaction
		, log_mean_func
	))
NIG_epg_y1_allages = NIG_epg_y1_allages.rename(columns={
	"<lambda_0>":"Prevalence", "log_NB_func":"Log Dispersion"
	,"g_interaction":"Log Gemometric Interaction"
	,"a_interaction":"Log Arithmetic Interaction"
	,"log_mean_func":"Log Mean Intensity"
	})
NIG_epg_y1_5_8 = scoreNIG[(scoreNIG["Study_Year"]==1) 
& (scoreNIG["X_Sect"]=='5_8')].pivot_table(
	values="Mean_eggs",index="Village_ID"
	,aggfunc = (lambda x : np.mean(x>0),log_mean_func))
NIG_epg_y1_5_8 = NIG_epg_y1_5_8.rename(columns={
	"<lambda_0>":"Prevalence 5-8", "log_mean_func":"Log Mean Intensity 5-8"
	})
NIG_epg_y1_9_12 = scoreNIG[(scoreNIG["Study_Year"]==1) 
& (scoreNIG["X_Sect"]=='9_12')].pivot_table(
	values="Mean_eggs",index="Village_ID"
	,aggfunc = (lambda x : np.mean(x>0),log_mean_func))
NIG_epg_y1_9_12 = NIG_epg_y1_9_12.rename(columns={
	"<lambda_0>":"Prevalence 9-12", "log_mean_func":"Log Mean Intensity 9-12"
	})
NIG_epg_y1 = pd.merge(NIG_epg_y1_allages,NIG_epg_y1_5_8
	,how='left',on='Village_ID').merge(NIG_epg_y1_9_12,how='left',on='Village_ID')

# Calculate for Mozambique
MOZ_epg_y1_allages = scoreMOZ[scoreMOZ["Study_Year"]==1].pivot_table(
	values="Mean_eggs",index="Village_ID"
	,aggfunc = ( log_NB_func
		, lambda x : np.mean(x>0)
		, g_interaction
		, a_interaction
		, log_mean_func
	))
MOZ_epg_y1_allages = MOZ_epg_y1_allages.rename(columns={
	"<lambda_0>":"Prevalence", "log_NB_func":"Log Dispersion"
	,"g_interaction":"Log Gemometric Interaction"
	,"a_interaction":"Log Arithmetic Interaction"
	,"log_mean_func":"Log Mean Intensity"
	})
MOZ_epg_y1_5_8 = scoreMOZ[(scoreMOZ["Study_Year"]==1) 
& (scoreMOZ["X_Sect"]=='5_8')].pivot_table(
	values="Mean_eggs",index="Village_ID"
	,aggfunc = (lambda x : np.mean(x>0),log_mean_func))
MOZ_epg_y1_5_8 = MOZ_epg_y1_5_8.rename(columns={
	"<lambda_0>":"Prevalence 5-8", "log_mean_func":"Log Mean Intensity 5-8"
	})
MOZ_epg_y1_9_12 = scoreMOZ[(scoreMOZ["Study_Year"]==1) 
& (scoreMOZ["X_Sect"]=='9_12')].pivot_table(
	values="Mean_eggs",index="Village_ID"
	,aggfunc = (lambda x : np.mean(x>0),log_mean_func))
MOZ_epg_y1_9_12 = MOZ_epg_y1_9_12.rename(columns={
	"<lambda_0>":"Prevalence 9-12", "log_mean_func":"Log Mean Intensity 9-12"
	})
MOZ_epg_y1 = pd.merge(MOZ_epg_y1_allages,MOZ_epg_y1_5_8
	,how='left',on='Village_ID').merge(MOZ_epg_y1_9_12,how='left',on='Village_ID')

# Calculate for mansoni countries
Sm_epg_y1_allages = scoreSm[scoreSm["Study_Year"]==1].pivot_table(
	values="Mean_epg",index="Village_ID"
	,aggfunc = ( log_NB_func
		, lambda x : np.mean(x>0)
		, gmean_barenbold_func
		, g_interaction
		, a_interaction
		, log_mean_func
	))
Sm_epg_y1_allages = Sm_epg_y1_allages.rename(columns={
	"<lambda_0>":"Prevalence", "log_NB_func":"Log Dispersion"
	,"gmean_barenbold_func":"Barenbold Geometric Mean"
	,"g_interaction":"Log Gemometric Interaction"
	,"a_interaction":"Log Arithmetic Interaction"
	,"log_mean_func":"Log Mean Intensity"
	})
Sm_epg_y1_5_8 = scoreSm[(scoreSm["Study_Year"]==1) 
& (scoreSm["X_Sect"]=='5_8')].pivot_table(
	values="Mean_epg",index="Village_ID"
	,aggfunc = (lambda x : np.mean(x>0)
	     ,gmean_barenbold_func
		 ,log_mean_func))
Sm_epg_y1_5_8 = Sm_epg_y1_5_8.rename(columns={
	"<lambda_0>":"Prevalence 5-8", "log_mean_func":"Log Mean Intensity 5-8"
	,"gmean_barenbold_func":"Barenbold Geometric Mean 5-8"
	})
Sm_epg_y1_9_12 = scoreSm[(scoreSm["Study_Year"]==1) 
& (scoreSm["X_Sect"]=='9_12')].pivot_table(
	values="Mean_epg",index="Village_ID"
	,aggfunc = (lambda x : np.mean(x>0)
	     ,gmean_barenbold_func
		 ,log_mean_func))
Sm_epg_y1_9_12 = Sm_epg_y1_9_12.rename(columns={
	"<lambda_0>":"Prevalence 9-12", "log_mean_func":"Log Mean Intensity 9-12"
	,"gmean_barenbold_func":"Barenbold Geometric Mean 9-12"
	})
Sm_epg_y1 = pd.merge(Sm_epg_y1_allages,Sm_epg_y1_5_8
	,how='left',on='Village_ID').merge(Sm_epg_y1_9_12,how='left',on='Village_ID')

# Merge egg data into overall dataframe
df = df.merge(pd.concat((NIG_epg_y1,MOZ_epg_y1,Sm_epg_y1)),how='left',on='Village_ID')

####################### Testing and 'True' prevalence ###########################

# Sh true prevalence using linear model of urine filtration values from Barenbold 2020 data
slope, intercept = 1.5446419579961295, 0.07487611247037279
# for Niger
NIG_true_prevalence_y1_allages = np.minimum(1,slope*NIG_epg_y1_allages['Prevalence']+intercept)
NIG_true_prevalence_y1_5_8 = np.minimum(1,slope*NIG_epg_y1_5_8['Prevalence 5-8']+intercept)
NIG_true_prevalence_y1_9_12 = np.minimum(1,slope*NIG_epg_y1_9_12['Prevalence 9-12']+intercept)
NIG_true_prevalence_y1 = pd.merge(NIG_true_prevalence_y1_allages,NIG_true_prevalence_y1_5_8,on='Village_ID').merge(NIG_true_prevalence_y1_9_12,on='Village_ID')
NIG_true_prevalence_y1.columns = ['True Prevalence','True Prevalence 5-8','True Prevalence 9-12']
NIG_true_prevalence_y1["Village_ID"] = NIG_true_prevalence_y1.index

# for Mozambique
MOZ_true_prevalence_y1_allages = np.minimum(1,slope*MOZ_epg_y1_allages['Prevalence']+intercept)
MOZ_true_prevalence_y1_5_8 = np.minimum(1,slope*MOZ_epg_y1_5_8['Prevalence 5-8']+intercept)
MOZ_true_prevalence_y1_9_12 = np.minimum(1,slope*MOZ_epg_y1_9_12['Prevalence 9-12']+intercept)
MOZ_true_prevalence_y1 = pd.merge(MOZ_true_prevalence_y1_allages,MOZ_true_prevalence_y1_5_8,on='Village_ID').merge(MOZ_true_prevalence_y1_9_12,on='Village_ID')
MOZ_true_prevalence_y1.columns = ['True Prevalence','True Prevalence 5-8','True Prevalence 9-12']
MOZ_true_prevalence_y1["Village_ID"] = MOZ_true_prevalence_y1.index

# Sm true prevalence using Barenbold 2021

# function to classify test type based on slides taken.
# if one test slide seems to be missing find closest standard pattern
def test_class(srs):
	if np.all(srs[['sm1b','sm1a','sm2b','sm3b','sm2a','sm3a']]!=' '):
		return('3dx2s')
	elif (np.all(srs[['sm1a','sm1b','sm2a','sm2b']]!=' ')&np.all(srs[['sm3b','sm3a']]==' '))|(np.all(srs[['sm1a','sm1b','sm3a','sm3b']]!=' ')&np.all(srs[['sm2b','sm2a']]==' '))|(np.all(srs[['sm2a','sm2b','sm3a','sm3b']]!=' ')&np.all(srs[['sm1b','sm1a']]==' ')):
		return('2dx2s')
	elif np.all(srs[['sm1b','sm1a']]!=' ')&np.all(srs[['sm2b','sm3b','sm2a','sm3a']]==' ')|(np.all(srs[['sm3b','sm3a']]!=' ')&np.all(srs[['sm2a','sm2b','sm1a','sm1b']]==' '))|(np.all(srs[['sm2b','sm2a']]!=' ')&np.all(srs[['sm3a','sm3b','sm1a','sm1b']]==' ')):
		return('1dx2s')
	elif np.sum(srs[['sm1b','sm1a','sm2b','sm3b','sm2a','sm3a']]!=' ')==1:
		return('1dx1s')
	elif np.sum(srs[['sm1b','sm1a','sm2b','sm3b','sm2a','sm3a']]!=' ')==5:
		return('3dx2s')
	elif np.sum(srs[['sm1b','sm1a','sm2b','sm3b','sm2a','sm3a']]!=' ')==3:
		return('2dx2s')
	else:
		return('unclassified')

# Barenbold et al. 2021 formula for prevalence assuming N=50
def true_prevalence(po,EPS,test_type):
	if test_type=='1dx1s':
		a0, a1, a2 = -7.19, 8.98, 13.54
	if test_type=='1dx2s':
		a0, a1, a2 = -6.24, 8.45, 13.51
	elif test_type=='2dx2s':
		a0, a1, a2 = -5.09, 8.06, 13.52
	elif test_type=='3dx2s':
		a0, a1, a2 = -4.12, 7.57, 13.29
	pt = po/(expit(a0+a1*(((EPS+0.01)/8)**(1/a2))))
	return(pt)

# Assign test type to each Sm village
for i in scoreSm.index:
	scoreSm.loc[i,'Test_type'] = test_class(scoreSm.loc[i])

# scoreSm = pd.read_csv('scoreSm_w_Test_type.csv',index_col=0)
testtypeSm = scoreSm.pivot_table(
	index='Village_ID',columns='Study_Year',values='Test_type'
	,aggfunc=lambda x: x.value_counts().index[0] )

# For Study_Year==1, find the proportion of tests of each type in each village
# , and the corresponding 'true' prevalence
test_counts_y1 = pd.DataFrame(
	np.zeros((len(scoreSm["Village_ID"].unique()),4))
	,columns=["3dx2s","2dx2s","1dx2s","1dx1s"]
	,index=scoreSm["Village_ID"].unique())
test_prevs_y1_allages = test_counts_y1.copy()
test_prevs_y1_5_8 = test_counts_y1.copy()
test_prevs_y1_9_12 = test_counts_y1.copy()
for village in scoreSm.loc[scoreSm['Study_Year']==1,'Village_ID'].unique():
	village_tests = scoreSm.loc[
		(scoreSm['Study_Year']==1)&(scoreSm['Village_ID']==village)
		,'Test_type']
	tests = village_tests.value_counts()
	village_prev_allages = Sm_epg_y1_allages.loc[village,'Prevalence']
	village_GMI_allages = Sm_epg_y1_allages.loc[village,'Barenbold Geometric Mean']
	village_prev_5_8 = Sm_epg_y1_5_8.loc[village,'Prevalence 5-8']
	village_GMI_5_8 = Sm_epg_y1_5_8.loc[village,'Barenbold Geometric Mean 5-8']
	village_prev_9_12 = Sm_epg_y1_9_12.loc[village,'Prevalence 9-12']
	village_GMI_9_12 = Sm_epg_y1_9_12.loc[village,'Barenbold Geometric Mean 9-12']
	for testtype in ["3dx2s","2dx2s","1dx2s","1dx1s"]:
		test_prevs_y1_allages.loc[village,testtype] = true_prevalence(village_prev_allages,village_GMI_allages,testtype)
		test_prevs_y1_5_8.loc[village,testtype] = true_prevalence(village_prev_5_8,village_GMI_5_8,testtype)
		test_prevs_y1_9_12.loc[village,testtype] = true_prevalence(village_prev_9_12,village_GMI_9_12,testtype)
		if testtype in tests.index:
			test_counts_y1.loc[village,testtype] = tests[testtype]
test_props_y1 = test_counts_y1.div(test_counts_y1.sum(axis=1),axis=0)

# Calculate true prevalence for each Sm village
Sm_true_prevalence_y1_allages = pd.DataFrame(
	(scoreSm["Village_ID"].unique(),
	pd.DataFrame(test_prevs_y1_allages.values*test_props_y1.values
	,index=test_counts_y1.index).sum(axis=1))).transpose()
Sm_true_prevalence_y1_allages.columns=["Village_ID","True Prevalence"]
Sm_true_prevalence_y1_5_8 = pd.DataFrame(
	(scoreSm["Village_ID"].unique(),
	pd.DataFrame(test_prevs_y1_5_8.values*test_props_y1.values
	,index=test_counts_y1.index).sum(axis=1))).transpose()
Sm_true_prevalence_y1_5_8.columns=["Village_ID","True Prevalence 5-8"]
Sm_true_prevalence_y1_9_12 = pd.DataFrame(
	(scoreSm["Village_ID"].unique(),
	pd.DataFrame(test_prevs_y1_9_12.values*test_props_y1.values
	,index=test_counts_y1.index).sum(axis=1))).transpose()
Sm_true_prevalence_y1_9_12.columns=["Village_ID","True Prevalence 9-12"]
Sm_true_prevalence_y1 = pd.merge(Sm_true_prevalence_y1_allages
				 ,Sm_true_prevalence_y1_5_8
				 ,on="Village_ID").merge(Sm_true_prevalence_y1_9_12
			     ,on="Village_ID")

# Merge true prevalences into overall data frame
df = df.merge(pd.concat((NIG_true_prevalence_y1,MOZ_true_prevalence_y1,Sm_true_prevalence_y1)),how='left',on='Village_ID')

########################### Secondary variables ###########################

# useful function
def log_func(x):
	result = []
	for y in x:
		if y>0:
			result.append(np.log(y+1))
		else:
			result.append(0)
	return(result)

vegetation_data = pd.read_csv('Data/all_village_vegetation.csv')
vegetation_data = vegetation_data[[ID in df["Village_ID"].unique() 
for ID in vegetation_data["Village_ID"]]]
df = df.merge(vegetation_data[["Village_ID","NDVI"]],how='left',on='Village_ID')

water_distance_data = pd.read_csv('Data/all_village_water_distance.csv')
water_distance_data = water_distance_data[[ID in df["Village_ID"].unique() 
for ID in water_distance_data["Village_ID"]]]
water_distance_data["Log Water Distance"]\
 = log_func(water_distance_data["Water Distance"])
df = df.merge(water_distance_data[["Village_ID","Log Water Distance"]]
	,how='left',on='Village_ID')

precipitation_data = pd.read_csv('Data/all_village_precipitation.csv')
precipitation_data = precipitation_data[[ID in df["Village_ID"].unique() 
for ID in precipitation_data["Village_ID"]]]
df = df.merge(precipitation_data[["Village_ID","Precipitation"]]
	,how='left',on='Village_ID')

tmin_data = pd.read_csv('Data/all_village_tmin.csv')
tmin_data = tmin_data[[ID in df["Village_ID"].unique() 
for ID in tmin_data["Village_ID"]]]
df = df.merge(tmin_data[["Village_ID","Minimum Temperature"]]
	,how='left',on='Village_ID')

pop_density_data = pd.read_csv('Data/all_village_pop_density.csv')
pop_density_data = pop_density_data[[ID in df["Village_ID"].unique() 
for ID in pop_density_data["Village_ID"]]]
pop_density_data["Log Population Density"]\
 = log_func(pop_density_data["Population Density"])
df = df.merge(pop_density_data[["Village_ID","Log Population Density"]]
	,how='left',on='Village_ID')

DHS_data = pd.read_csv('Data/all_village_DHS.csv')
DHS_data = DHS_data[[ID in df["Village_ID"].unique()
for ID in DHS_data["Village_ID"]]]
df = df.merge(DHS_data[["Village_ID","Improved Water","Improved Sanitation","3rd Dose DPT","Education Female","Education Male","Stunting Under-5","Underweight Under-5","Mortality Under-5","Relative Wealth"]],how='left',on='Village_ID')

########################### Secondary-egg interaction ###########################

prevalence_density_data = pd.read_csv('Data/all_village_prevalence_density.csv')
prevalence_density_data = prevalence_density_data[[ID in df["Village_ID"].unique() 
	for ID in prevalence_density_data["Village_ID"]]]
prevalence_density_data["Log Prevalence Density"]\
	= log_func(prevalence_density_data["Prevalence Density"])
df = df.merge(prevalence_density_data[["Village_ID","Log Prevalence Density"]]
	,how='left',on='Village_ID')

########################### Outcomes ###########################

# haematobium moderate/heavy infection threshold is 50 eggs
NIG_outcomes = scoreNIG[scoreNIG["Study_Year"]==5].pivot_table(
	values="Mean_eggs",index="Village_ID"
	,aggfunc = (lambda x : np.mean(x>0)
		, lambda x : np.mean(x>50)
	))
NIG_outcomes = NIG_outcomes.rename(columns={
	"<lambda_0>":"Prevalence Outcome","<lambda_1>":"Intensity Outcome"
	})
NIG_outcomes['True Prevalence Outcome'] = np.minimum(1,slope*NIG_outcomes['Prevalence Outcome']+intercept)
NIG_outcomes['Village_ID']=NIG_outcomes.index

MOZ_outcomes = scoreMOZ[scoreMOZ["Study_Year"]==5].pivot_table(
	values="Mean_eggs",index="Village_ID"
	,aggfunc = (lambda x : np.mean(x>0)
		, lambda x : np.mean(x>50)
	))
MOZ_outcomes = MOZ_outcomes.rename(columns={
	"<lambda_0>":"Prevalence Outcome","<lambda_1>":"Intensity Outcome"
	})
MOZ_outcomes['True Prevalence Outcome'] = np.minimum(1,slope*MOZ_outcomes['Prevalence Outcome']+intercept)
MOZ_outcomes['Village_ID']=MOZ_outcomes.index

# mansoni moderate/heavy infection threshould is 100 epg
Sm_outcomes = scoreSm[scoreSm["Study_Year"]==5].pivot_table(
	values="Mean_epg",index="Village_ID"
	,aggfunc = (lambda x : np.mean(x>0)
	    , gmean_barenbold_func
		, lambda x : np.mean(x>100)
	))
Sm_outcomes = Sm_outcomes.rename(columns={
	"<lambda_0>":"Prevalence Outcome"
	,"gmean_barenbold_func":"Barenbold Geometric Mean Outcome"
	,"<lambda_1>":"Intensity Outcome"
	})

# For Study_Year==5, find the proportion of tests of each type in each village
# , and the corresponding 'true' prevalence
test_counts_y5 = pd.DataFrame(
	np.zeros((len(scoreSm["Village_ID"].unique()),4))
	,columns=["3dx2s","2dx2s","1dx2s","1dx1s"]
	,index=scoreSm["Village_ID"].unique())
test_prevs_y5 = test_counts_y5.copy()
for village in scoreSm.loc[scoreSm['Study_Year']==5,'Village_ID'].unique():
	village_tests = scoreSm.loc[
		(scoreSm['Study_Year']==1)&(scoreSm['Village_ID']==village)
		,'Test_type']
	tests = village_tests.value_counts()
	village_prev_allages = Sm_outcomes.loc[village,'Prevalence Outcome']
	village_GMI_allages = Sm_outcomes.loc[village,'Barenbold Geometric Mean Outcome']
	for testtype in ["3dx2s","2dx2s","1dx2s","1dx1s"]:
		test_prevs_y5.loc[village,testtype] = true_prevalence(village_prev_allages,village_GMI_allages,testtype)
		if testtype in tests.index:
			test_counts_y5.loc[village,testtype] = tests[testtype]
test_props_y5 = test_counts_y5.div(test_counts_y5.sum(axis=1),axis=0)

# Calculate true prevalence for each Sm village
Sm_true_prevalence_y5 = pd.DataFrame(
	(test_counts_y5.index,
	pd.DataFrame(test_prevs_y5.values*test_props_y5.values
	).sum(axis=1))).transpose()
Sm_true_prevalence_y5.columns=["Village_ID","True Prevalence Outcome"]

Sm_outcomes_w_true = Sm_outcomes.merge(Sm_true_prevalence_y5,on='Village_ID')
outcomes = pd.concat((NIG_outcomes,MOZ_outcomes,Sm_outcomes_w_true))
df = df.merge(outcomes,how='left',on='Village_ID')

df["Relative Outcome"]\
= (np.array(df["Prevalence"])-np.array(df["Prevalence Outcome"]))\
/np.maximum(np.array(df["Prevalence"]),np.array(df["Prevalence Outcome"]))
df["True Relative Outcome"]\
= (np.array(df["True Prevalence"])-np.array(df["True Prevalence Outcome"]))\
/np.maximum(np.array(df["True Prevalence"]),np.array(df["True Prevalence Outcome"]))

df["Prevalence Hotspot"] = df["Prevalence Outcome"]>.1
df["Intensity Hotspot"] = df["Intensity Outcome"]>.01
df["Relative Hotspot"] = df["Relative Outcome"]<.35
df["True Prevalence Hotspot"] = df["True Prevalence Outcome"]>.1
df["True Relative Hotspot"] = df["True Relative Outcome"]<.35
# the only NaNs should be places that start and end with zero prevalence
# these should not be counted as hotspots
df["Relative Outcome"] = df["Relative Outcome"].fillna(0)
df["Relative Hotspot"] = df["Relative Hotspot"].fillna(False)

df["Prevalence Relative Hotspot"] = (df["Prevalence Hotspot"] & df["Relative Hotspot"])
df["Prevalence Intensity Hotspot"] = (df["Prevalence Hotspot"] & df["Intensity Hotspot"])
df["True Prevalence Relative Hotspot"] = (df["True Prevalence Hotspot"] & df["True Relative Hotspot"])
df["True Prevalence Intensity Hotspot"] = (df["True Prevalence Hotspot"] & df["Intensity Hotspot"])

########################### Save dataframe ###########################

df.to_csv('Data/all_hotspot_prediction_data.csv')
