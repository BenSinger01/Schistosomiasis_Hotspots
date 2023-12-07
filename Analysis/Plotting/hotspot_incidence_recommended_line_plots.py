import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc
import sys

filename, folder = sys.argv[1], sys.argv[2]

plt.rcParams['font.size'] = '11'
rc('font',**{'family':'serif','serif':['Times']})
rc('text',usetex=True)

x = np.linspace(0,1,1000)

scores = pd.read_csv(filename)
pSh_sens = scores.loc[(scores["Model"]=="LinearRegression")&
                 (scores["Trainset"]=="Sh")&
                 (scores["Outcome"]=="Prevalence")&
                 (scores["Metric"]=="Sensitivity"),"Score"].values[0]
pSh_spec = scores.loc[(scores["Model"]=="LinearRegression")&
                 (scores["Trainset"]=="Sh")&
                 (scores["Outcome"]=="Prevalence")&
                 (scores["Metric"]=="Specificity"),"Score"].values[0]
pSm_sens = scores.loc[(scores["Model"]=="ElasticNetLogistic")&
                 (scores["Trainset"]=="Sm")&
                 (scores["Outcome"]=="Prevalence")&
                 (scores["Metric"]=="Sensitivity"),"Score"].values[0]
pSm_spec = scores.loc[(scores["Model"]=="ElasticNetLogistic")&
                 (scores["Trainset"]=="Sm")&
                 (scores["Outcome"]=="Prevalence")&
                 (scores["Metric"]=="Specificity"),"Score"].values[0]

iSh_sens = scores.loc[(scores["Model"]=="GradientBoostingClassifier")&
                 (scores["Trainset"]=="Sh")&
                 (scores["Outcome"]=="Prevalence Intensity")&
                 (scores["Metric"]=="Sensitivity"),"Score"].values[0]
iSh_spec = scores.loc[(scores["Model"]=="GradientBoostingClassifier")&
                 (scores["Trainset"]=="Sh")&
                 (scores["Outcome"]=="Prevalence Intensity")&
                 (scores["Metric"]=="Specificity"),"Score"].values[0]
iSm_sens = scores.loc[(scores["Model"]=="RandomForestClassifier")&
                 (scores["Trainset"]=="Sm")&
                 (scores["Outcome"]=="Prevalence Intensity")&
                 (scores["Metric"]=="Sensitivity"),"Score"].values[0]
iSm_spec = scores.loc[(scores["Model"]=="RandomForestClassifier")&
                 (scores["Trainset"]=="Sm")&
                 (scores["Outcome"]=="Prevalence Intensity")&
                 (scores["Metric"]=="Specificity"),"Score"].values[0]

pPPVSh = pSh_sens*x/(pSh_sens*x+(1-pSh_spec)*(1-x))
pNPVSh = pSh_spec*(1-x)/((1-pSh_sens)*x+pSh_spec*(1-x))

pPPVSm = pSm_sens*x/(pSm_sens*x+(1-pSm_spec)*(1-x))
pNPVSm = pSm_spec*(1-x)/((1-pSm_sens)*x+pSm_spec*(1-x))

iPPVSh = iSh_sens*x/(iSh_sens*x+(1-iSh_spec)*(1-x))
iNPVSh = iSh_spec*(1-x)/((1-iSh_sens)*x+iSh_spec*(1-x))

iPPVSm = iSm_sens*x/(iSm_sens*x+(1-iSm_spec)*(1-x))
iNPVSm = iSm_spec*(1-x)/((1-iSm_sens)*x+iSm_spec*(1-x))


fig,ax=plt.subplots(2,2,sharey=True,sharex=False,figsize=(6.5,7))

ax[0,0].plot(x*100,pPPVSh,'#DC267F',label='Positive')
ax[0,0].set_ylabel('Predictive value')
ax[0,0].set_title(r'$\it{S.\ haematobium}$ prevalence hotspot')
ax[1,0].set_title(r'$\it{S.\ haematobium}$ intensity hotspot')
ax[0,1].plot(x*100,pPPVSm,'#DC267F')
ax[0,1].set_title(r'$\it{S.\ mansoni}$ prevalence hotspot')
ax[1,1].set_title(r'$\it{S.\ mansoni}$ intensity hotspot')
ax[0,0].plot(x*100,pNPVSh,'#648FFF',label='Negative')
ax[0,1].plot(x*100,pNPVSm,'#648FFF')
ax[0,0].legend()
# ax[0,0].axvline(.242857,c='k',linestyle='--')
# ax[0,0].axvline(.75,c='k',linestyle='--')
# ax[0,1].axvline(.410256,c='k',linestyle='--')
# ax[0,1].axvline(.512690,c='k',linestyle='--')
# ax[0,1].axvline(.845528,c='k',linestyle='--')
ax[1,0].plot(x*100,iPPVSh,'#DC267F',label='Positive')
ax[1,0].set_ylabel('Predictive value')
ax[1,1].plot(x*100,iPPVSm,'#DC267F')
ax[1,0].plot(x*100,iNPVSh,'#648FFF',label='Negative')
ax[1,0].set_xlabel('Hotspot prevalence')
ax[0,0].set_xlabel('Hotspot prevalence')
ax[0,1].set_xlabel('Hotspot prevalence')
ax[1,1].plot(x*100,iNPVSm,'#648FFF')
ax[1,1].set_xlabel('Hotspot prevalence')

ax[0,0].text(-0.25,1.15,'A',transform=ax[0,0].transAxes,fontsize=16,fontweight='bold',va='top',ha='right')
ax[1,0].text(-0.25,1.15,'B',transform=ax[1,0].transAxes,fontsize=16,fontweight='bold',va='top',ha='right')

plt.tight_layout()
plt.savefig(folder+"Figures/hotspot_incidence_recommended_line_plots.png")
