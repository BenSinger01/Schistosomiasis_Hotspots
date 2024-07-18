# Development of prediction models to identify hotspots of schistosomiasis in endemic regions to guide mass drug administration

## Summary
This repository contains analytic code related our study on statistical models to predict hotspots of schistosomiasis. This code implements statistical models to predict hotspots at baseline prior to treatment comparing three common hotspot definitions, using epidemiologic, survey-based, and remote sensing data. The analysis also includes model evaluation according to various metrics, including three alternate ways to construct the test set. Finally, we include code to reproduce the figures and tables in the associated publication.

The data used for this analysis is available through various public repositories, some of which require a formal request for data. Therefore we cannot make all data for this analysis available through this repository. Please see the associated publication for more information on the data sources used.

The study is now published in [PNAS](https://doi.org/10.1073/pnas.2315463120).

The full citation for the article is: Benjamin J. Singer, Jean T. Coulibaly, Hailey J. Park, Jason R. Andrews, Isaac I. Bogoch, and Nathan C. Lo. Development of prediction models to identify hotspots of schistosomiasis in endemic regions to guide mass drug administration. Proceedings of the National Academy of Sciences, 121 (2) e2315463120, 2024. doi: 10.1073/pnas.2315463120

This code was written and tested using Python version 3.9.7 and Scikit-learn version 1.1.1.

## Structure 
**Data** – Empty placeholder directory. We cannot make available the raw data required for full replication due to licensing issues. Please contact us for more information, or submit an issue.

**Analysis** – Directory containing analyitic code in Python.

- **Data_Processing** - Code to process, sort, and scale data.

- **Models** - Code to train statistical models of schistosomiasis hotspots.

- **Plotting** - Code to generate figures and tables in the publication.

**Outputs** – Directory containing trained models.

**Figures** – Directory containing figures and tables.

## Running the Main Analysis 
RUN_ANALYSIS.bash contains instructions to run the full analysis. When the Data directory is populated, this script sorts data, trains models, evaluates models, and generates tables and figures.

## Contact
Please direct any questions to the study authors:
Ben Singer, Stanford University, contact: bjsinger@stanford.edu
Nathan Lo, Stanford University, contact: nathan.lo@stanford.edu
