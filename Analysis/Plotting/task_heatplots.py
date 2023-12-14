import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc
import sys

filename, metric, model, folder, var, validation_approach = sys.argv[1:]

plt.rcParams['font.size'] = '11'
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text',usetex=True)

n_KEN = pd.read_csv(folder+"Data/"+validation_approach+"Test_Data/KEN.csv").shape[0]
n_TAN = pd.read_csv(folder+"Data/"+validation_approach+"Test_Data/TAN.csv").shape[0]
n_KENTAN = pd.read_csv(folder+"Data/"+validation_approach+"Test_Data/all_COT.csv").shape[0]
n_TANCOT = pd.read_csv(folder+"Data/"+validation_approach+"Test_Data/all_KEN.csv").shape[0]
n_COTKEN = pd.read_csv(folder+"Data/"+validation_approach+"Test_Data/all_TAN.csv").shape[0]

scores = pd.read_csv(filename)
scores = scores[scores['Metric']==metric]
if model != 'Max':
	scores = scores[scores['Model']==model]
WCSh=scores[scores['Trainset']=='NIG']
CCSh=scores[scores['Trainset']=='Sh']
BCSh=scores[scores['Trainset']=='all_NIG']
KEN = scores[scores['Trainset']=='KEN']
TAN = scores[scores['Trainset']=='TAN']
WCSm = KEN.copy()
WCSm['Score'] = (n_KEN*np.array(KEN['Score'])+n_TAN*np.array(TAN['Score']))/(n_KEN+n_TAN)
CCSm=scores[scores['Trainset']=='Sm']
all_KENTAN = scores[scores['Trainset']=='all_KENTAN']
all_TANCOT = scores[scores['Trainset']=='all_TANCOT']
all_COTKEN = scores[scores['Trainset']=='all_COTKEN']
BCSm = all_KENTAN.copy()
BCSm['Score'] = (n_KENTAN*np.array(all_KENTAN['Score'])+n_TANCOT*np.array(all_TANCOT['Score'])+n_COTKEN*np.array(all_COTKEN['Score']))/(n_KENTAN+n_TANCOT+n_COTKEN)

Sh = pd.concat((CCSh,BCSh,WCSh)).pivot_table(
	values='Score',index='Outcome',columns='Trainset',aggfunc='max').reindex(
			["Prevalence","Prevalence Intensity","Prevalence Relative"])
Sm = pd.concat((CCSm,BCSm,WCSm)).pivot_table(
	values='Score',index='Outcome',columns='Trainset',aggfunc='max').reindex(
			["Prevalence","Prevalence Intensity","Prevalence Relative"])

Sh = Sh[['Sh','all_NIG','NIG']]
Sm = Sm[['Sm','all_KENTAN','KEN']]

fig, ax = plt.subplots(1,2,sharey=True,figsize=(8.5,5.5))

ax[0].imshow(Sh,vmin=0.47,vmax=0.93)
ax[0].set_title(r'$\it{S.\ haematobium}$')
ax[0].set_xticks([0,1,2])
ax[0].set_xticklabels(['Combined\ncountries','Between\ncountries','Within\ncountry'])
ax[0].set_yticks([0,1,2])
ax[0].set_yticklabels(['Prevalence\nhotspot','Intensity\nhotspot','Relative\nhotspot'])
ax[0].set_ylabel('Hotspot definition')
image = ax[1].imshow(Sm,vmin=0.47,vmax=0.93)
ax[1].set_xticks([0,1,2])
ax[1].set_xticklabels(['Combined\ncountries','Between\ncountries','Within\ncountry',])
ax[1].set_title(r'$\it{S.\ mansoni}$')
plt.tight_layout()

for species in range(2):
	for outcome in range(3):
	    for testset in range(3):
        	text = ax[species].text(testset, outcome, "{:.2f}".format([Sh,Sm][species].iloc[outcome, testset]),
                       	ha="center", va="center", color="w")


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
plt.colorbar(image,cax=cbar_ax,label='Balanced accuracy')
fig.text(0.465, 0.1, 'Test set', ha='center')
plt.savefig(folder+"Figures/"+var+validation_approach+'task_heatplots_'+metric+model+'.png',dpi=300)
