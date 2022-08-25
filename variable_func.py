def variable_func(dataset):
	if dataset in ["KEN","TAN","COT","Sm","all_COTKEN","all_KENTAN","all_TANCOT"]:
		return(["SAC Rounds","Community Rounds","Prevalence"
			,"Log Gemometric Interaction","Log Arithmetic Interaction","Log Dispersion"
			,"Log Mean Intensity","Prevalence 5-8","Log Mean Intensity 5-8"
			,"Prevalence 9-12","Log Mean Intensity 9-12","NDVI","Log Water Distance"
			,"Precipitation","Minimum Temperature","Log Population Density"
			,"Log Prevalence Density"])
	elif dataset in ["NIG","MOZ","Sh","all_NIG"]:
		return(["SAC Rounds","Community Rounds","Prevalence"
			,"Log Gemometric Interaction","Log Arithmetic Interaction","Log Dispersion"
			,"Log Mean Intensity","Prevalence 5-8","Log Mean Intensity 5-8"
			,"Prevalence 9-12","Log Mean Intensity 9-12","Log Water Distance"
			,"Minimum Temperature","Log Population Density"
			,"Log Prevalence Density"])

def alt_variable_func(dataset):
	return(["SAC Rounds","Community Rounds","Prevalence"
		,"Log Gemometric Interaction","Log Arithmetic Interaction","Log Dispersion"
		,"Log Mean Intensity","Prevalence 5-8","Log Mean Intensity 5-8"
		,"Prevalence 9-12","Log Mean Intensity 9-12"])

def dummy_variable_func(dataset):
	if dataset in ["KEN","TAN","COT","Sm","all_COTKEN","all_KENTAN","all_TANCOT"]:
		return(["Dummy_KEN","Dummy_TAN","Dummy_COT","SAC Rounds","Community Rounds","Prevalence"
			,"Log Gemometric Interaction","Log Arithmetic Interaction","Log Dispersion"
			,"Log Mean Intensity","Prevalence 5-8","Log Mean Intensity 5-8"
			,"Prevalence 9-12","Log Mean Intensity 9-12","NDVI","Log Water Distance"
			,"Precipitation","Minimum Temperature","Log Population Density"
			,"Log Prevalence Density"])
	elif dataset in ["NIG","MOZ","Sh","all_NIG"]:
		return(["Dummy_MOZ","Dummy_NIG","SAC Rounds","Community Rounds","Prevalence"
			,"Log Gemometric Interaction","Log Arithmetic Interaction","Log Dispersion"
			,"Log Mean Intensity","Prevalence 5-8","Log Mean Intensity 5-8"
			,"Prevalence 9-12","Log Mean Intensity 9-12","Log Water Distance"
			,"Minimum Temperature","Log Population Density"
			,"Log Prevalence Density"])

def dummy_only_variable_func(dataset):
	if dataset in ["KEN","TAN","COT","Sm","all_COTKEN","all_KENTAN","all_TANCOT"]:
		return(["Dummy_KEN","Dummy_TAN","Dummy_COT"])
	elif dataset in ["NIG","MOZ","Sh","all_NIG"]:
		return(["Dummy_MOZ","Dummy_NIG"])