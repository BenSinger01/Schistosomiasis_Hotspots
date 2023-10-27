import rasterio as rio
import matplotlib.pyplot as plt
import pandas as pd
from rasterio.plot import show
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import time

class Country:

    def __init__(self, name, code, bounds):
        self.name = name
        self.code = code
        self.min_lat = bounds[0]
        self.max_lat = bounds[1]
        self.min_lon = bounds[2]
        self.max_lon = bounds[3]

kenya = Country('kenya', 'KEN', [-1, 0, 34, 36])
tanzania = Country('tanzania', 'TAN', [-3, -2, 31.5, 34])
niger = Country('niger', 'NIG', [12.5, 14.5, 0.5, 4])
mozambique = Country('mozam', 'MOZ', [-14.2, -11.7, 38, 40.6])
ivory = Country('ivory', 'COT', [6.5, 8, -8.5,-6.5])

water_file = "Data/DHS_Data/IHME_LMIC_WASH_2000_2017_W_IMP_PERCENT_MEAN_2011_Y2020M06D02.TIF"
sanitation_file = "Data/DHS_Data/IHME_LMIC_WASH_2000_2017_S_IMP_PERCENT_MEAN_2011_Y2020M06D02.TIF"
dpt_file = "Data/DHS_Data/IHME_AFRICA_DPT_2000_2016_DPT3_COVERAGE_PREV_MEAN_2011_Y2019M04D01.TIF"
f_ed_file = "Data/DHS_Data/IHME_LMIC_EDU_2000_2017_MEAN_15_49_FEMALE_MEAN_Y2019M12D24.TIF"
m_ed_file = "Data/DHS_Data/IHME_LMIC_EDU_2000_2017_MEAN_15_49_MALE_MEAN_Y2019M12D24.TIF"
stunt_file = "Data/DHS_Data/IHME_GLOBAL_CGF_2000_2019_STUNTING_PREV_PERCENT_A1_S3_MEAN_2011_Y2020M08D31.TIF"
uw_file = "Data/DHS_Data/IHME_GLOBAL_CGF_2000_2019_UNDERWEIGHT_PREV_PERCENT_A1_S3_MEAN_2011_Y2020M08D31.TIF"
mortality_file = "Data/DHS_Data/IHME_LMICS_U5M_2000_2017_Q_UNDER5_MEAN_Y2019M10D16.TIF"

files = {'W_IMP': water_file, 'S_IMP':sanitation_file, 'DPT3':dpt_file,'EDU_F':f_ed_file,
         'EDU_M':m_ed_file,'STUNTING':stunt_file,'UNDERWEIGHT':uw_file, 'Q_UNDER5':mortality_file}

### EXTRACTING DATA ###
def extract_data(country, var):
    name = country.code
    with rio.open('DHS_Data/'+var+'.TIF') as dataset:
        if (var[:3] == 'EDU')|(var == 'Q_UNDER5'):
            val = dataset.read(12) # select 2011 when multiple years in data
        else:
            val = dataset.read(1) # select band 1 otherwise
        no_data=dataset.nodata
        [height, width] = val.shape
        affine = dataset.transform
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rio.transform.xy(affine, rows, cols)
        lon = np.array(xs)
        lat = np.array(ys)
        val_mask = val!=no_data
        lon_mask = (lon>=country.min_lon)&(lon<=country.max_lon)
        lat_mask = (lat>=country.min_lat)&(lat<=country.max_lat)
        mask = (val_mask&lon_mask)&lat_mask
        ds = pd.DataFrame({"longitude":lon[mask],'latitude':lat[mask],"data":val[mask]})
        return ds


### MATCHING DATA TO VILLAGES ###
def match_villages(v_coords,data):
    v_lats = np.radians(np.array(v_coords['Latitude']))
    v_lons = np.radians(np.array(v_coords['Longitude']))
    d_lats = np.radians(np.array(data['latitude'])[:,None])
    d_lons = np.radians(np.array(data['longitude'])[:,None])
    lat_diffs = v_lats-d_lats
    lon_diffs = v_lons-d_lons
    d = np.sin(lat_diffs/2)**2 + np.cos(v_lats)*np.cos(d_lats) * np.sin(lon_diffs/2)**2
    dists = 2 * 6378.1 * np.arcsin(np.sqrt(d))
    argmins = np.argmin(dists,axis=0)
    mins = [dists[argmins[i],i] for i in range(len(argmins))]
    matched_data = data.iloc[argmins]
    matched_data.index = v_coords['Village_ID']
    df = v_coords.merge(matched_data, on='Village_ID')
    df.set_index('Village_ID', inplace=True)
    df["Distances"] = mins
    df.rename(columns={'Unnamed: 0':'Pixel No.','longitude':'Pixel Longitude','latitude':'Pixel Latitude'}, inplace=True)
    return(df)


## Process data
vdata = pd.read_csv('all_village_coords.csv')

countries = [kenya,tanzania,ivory,niger,mozambique]
variables = ['W_IMP','S_IMP','DPT3','EDU_F','EDU_M','STUNTING','UNDERWEIGHT','Q_UNDER5']

for country in countries:
    for v in variables:
        print(country.code, v)
        start_time = time.time()
        data = extract_data(country, v)
        data.to_csv('DHS_Data/'+country.code+'_'+v+'.csv')
        print("--- %s seconds ---" % (time.time() - start_time))

# # Find village matches
# for country in countries:
#     for v in variables:
#         cf_data = pd.read_csv('DHS_Data/'+country.code+'_'+v+'.csv')
#         start_time = time.time()
#         matched_cf_data = match_villages(vdata[vdata['Village_ID'].str.startswith(country.code)], cf_data)
#         print("--- %s seconds ---" % (time.time() - start_time))
#         matched_cf_data.to_csv('DHS_Data/'+country.code+'_'+v+'_matched.csv')

# # Concatenate matching 
# for v in variables:
#     df = pd.DataFrame()
#     for country in countries:
#         c_data = pd.read_csv('DHS_Data/'+country.code+'_'+v+'_matched.csv')
#         df = pd.concat([df, c_data])
#     df.to_csv('DHS_Data/'+v+'_matched.csv')

for country in countries:
    print(country.code)
    df = vdata[vdata['Village_ID'].str.startswith(country.code)]
    for v in variables:
        print(v)
        data = pd.read_csv('DHS_Data/'+country.code+'_'+v+'.csv')
        matched_data = match_villages(df, data)['data']
        matched_data.rename(v, inplace=True)
        df = df.merge(matched_data,on='Village_ID')
    data = pd.read_csv('DHS_Data/'+country.code+'_RWI.csv')
    matched_data = match_villages(df, data)['rwi']
    matched_data.rename('RWI', inplace=True)
    df = df.merge(matched_data,on='Village_ID')
    df.to_csv('DHS_Data/'+country.code+'.csv')

ken_df = pd.read_csv('DHS_Data/KEN.csv')
tan_df = pd.read_csv('DHS_Data/TAN.csv')
nig_df = pd.read_csv('DHS_Data/NIG.csv')
moz_df = pd.read_csv('DHS_Data/MOZ.csv')
cot_df = pd.read_csv('DHS_Data/COT.csv')

df = pd.concat([ken_df, tan_df, nig_df, moz_df, cot_df])

# print(df)

data_df=df.rename(columns={'W_IMP':'Improved Water','S_IMP':'Improved Sanitation', 
              'DPT3':'3rd Dose DPT','EDU_F':'Education Female','EDU_M':'Education Male'
              ,'STUNTING':'Stunting Under-5','UNDERWEIGHT':'Underweight Under-5',
              'Q_UNDER5':'Mortality Under-5','RWI':'Relative Wealth'})
print(data_df)

data_df.to_csv('DHS_Data/all_village_DHS_data.csv')

