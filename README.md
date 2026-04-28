# eco395m-ml-final-project
## Project Overview
The objective of this project is to predict forest fires by replicating a portion of _Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression_, and generate new models to compete with the fire predictions from the study. In this project we engineer new features, implement new models, and utilize chronos embeddings. 
## Data
### Data Sources
The project analyzes the "US Wildfire Dataset (2014-2025)" found on [kaggle](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset/data). This data set was sourced from The dataset provides information on precipitation, relative humidity, specific humidity, solar radiation, temperatures, wind speed, vapor pressure deficit, fuel moisture indices, energy release component, burning index, and evapotranspiration across 126,800 samples in the continental United States. This data is split into 75 day windows at each coordinate. Each 75 day window comprises of 60 days prior to the event, the ignition/non-ignition day, and the following 14 days. The data includes 50,720 positive (ignition) events, and 76,080 synthesized negative events. 

_Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression,
ICMLA 2025._
### Units & variables

Precipitation (pr): mm/day

Relative humidity (rmax/rmin): %

Specific humidity (sph): kg/kg

Solar radiation (srad): W/m²

Temperatures (tmmn/tmmx): °C

Wind speed (vs): m/s

Vapor pressure deficit (vpd): kPa

Fuel moisture indices (fm100/fm1000): %

Energy release component (erc): index

Burning index (bi): index

Evapotranspiration (etr, pet): mm/day

### Embeddings, Data Collection, and Cleaning
We utilize dstack to accelerate model training. We are using chronos forecasting with the pretrained Amazon Chronos T5 Mini model to extract feature embeddings. Then using those embeddings as features to improve our model performance.   

### Data Limitations


## Methodology
Firstly we extracted our wildfire data from Kaggle using the Kaggle API and from Weather Bug using the Weather Bug API.
Then we conducted train and test split using random split to follow methodology of _Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression_. 
From there we conducted exploratory data analysis, creating visulizations, distributions, and simple cleaning. 
In order to train the models we transformed the data into 75 x 15 matrices that include prior period, ignition, post period and model features.
We obtained baseline estimates of the following models: CNN-LSTM, XGBoost, Gradient Boosting, Random Forest, K-Nearest Neighbors, Simple-MLP, Decision Tree, Two-Layer-LSTM, LightTS-Inspired, Logistic Regression, Naive Bayes. 
Then we extracted feature embeddings using a pretrained chronos T5 Mini model for each of our features. From there we estimated the models with the new embeddings. 


### Modeling Limitations
## Results
## Recommendations
## Reproduction
1. Clone the repository git@github.com:Williamsillon/eco395m-ml-final-project.git
2. Install additional packages pip install -r requirements
