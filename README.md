# eco395m-ml-final-project
## Project Overview
The objective of this project is to predict forest fires by replicating a portion of _Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression_, and generate new models to compete with the fire predictions from the study. In this project we engineer new features, implement new models, and utilize chronos embeddings. 
## Data
### Data Sources
The project analyzes the "US Wildfire Dataset (2014-2025)" found on [kaggle](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset/data). The dataset provides information on precipitation, relative humidity, specific humidity, solar radiation, temperatures, wind speed, vapor pressure deficit, fuel moisture indices, energy release component, burning index, and evapotranspiration across 126,800 samples in the continental United States. This data is time series data split into 75 day windows at each coordinate. Each 75 day window comprises of 60 days prior to the event, the ignition/non-ignition day, and the following 14 days. The data includes 50,720 positive (ignition) events, and 76,080 synthesized negative events. 

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
We utilize dstack to accelerate our model training. We are using chronos forecasting with the pretrained Amazon Chronos T5 Mini model to extract feature embeddings. Then using those embeddings as features to improve our model performance. 

### Exploratory Data Analysis
The following visualizations are used to better understand the structure of the dataset and identify any patterns or inconsistencies before modeling. 

#### Correlation Matrix
<img src="artifacts/correlation_matrix.png" width="40%"/>

*Blayne??*

#### Feature Distributions
<img src="artifacts/feature_distributions.png" width="40%"/>

We plotted the distribution of all 15 weather and fire behavior features separated by wildfire and non-wildfire days. The distributions show that wildfire days tend to have higher values for burning index, energy release component, vapor pressure deficit, and temperature, and lower values for fuel moisture and relative humidity. This makes intuitive sense as hot, dry conditions with low fuel moisture are the physical drivers of wildfire ignition and spread.

#### Temporal Patterns
<img src="artifacts/temporal_patterns.png" width="40%"/>

We examined how six key features trend across the 75-day window for wildfire versus non-wildfire sequences, with the dashed line marking the approximate ignition day around day 60. The most notable finding is that wildfire sequences show persistently more extreme fire weather conditions well before the ignition event occurs, not just on the ignition day itself. Vapor Pressure Deficit and Energy Release Component are consistently elevated for wildfire sequences throughout the entire pre-ignition period, while Fuel Moisture and Min Relative Humidity are persistently lower. Max Temperature runs consistently hotter for wildfire sequences across the full window. Overall this confirms that the 60-day pre-ignition period carries strong predictive signal and motivates the use of time-series aware models that can capture how these conditions evolve over time rather than treating each day independently.

### Data Limitations
## Methodology
First, we extracted our wildfire data from Kaggle using the Kaggle API. Then, we conducted a train and test split using a random split to follow the methodology of Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression. From there, we conducted exploratory data analysis, creating visualizations, distributions, and performing basic data cleaning to better understand the structure of the dataset and identify any inconsistencies. In order to train the models, we transformed the data into 75x15 matrices that include prior period, ignition, post period, and model features. This allows us to incorporate both spatial and temporal dynamics of wildfire activity into the modeling framework. Our target variable is defined based on wildfire occurrence/intensity. All predictors are aligned to ensure consistency across time periods. We obtained baseline estimates of the following models: CNN-LSTM, XGBoost, Gradient Boosting, Random Forest, K-Nearest Neighbors, Simple-MLP, Decision Tree, Two-Layer-LSTM, LightTS-Inspired, Logistic Regression, and Naive Bayes. These models were chosen to capture a wide range of relationships in the data, including both linear and nonlinear patterns as well as temporal dependencies.

#### Baseline Model: Logistic Regression
A logistic regression model is used as a baseline benchmark. This provides a simple and interpretable reference point for evaluating more complex models. However, we expect this model to have limitations due to nonlinear relationships and interactions between environmental and weather-related predictors.

#### Tree-Based and Ensemble Models
Tree-based models such as Decision Trees, Random Forest, Gradient Boosting, and XGBoost are implemented to capture nonlinear relationships and higher-order interactions between predictors. Random Forest helps reduce variance through averaging across trees, while boosting-based methods like Gradient Boosting and XGBoost iteratively improve model performance by focusing on previous errors. These models are particularly useful given that  wildfire risk may depend on complex combinations of weather and environmental conditions. Since these models are not inherently time-series aware, the 3D sequence tensor is flattened to 1,125 features before modeling, meaning they cannot capture the temporal ordering of the 75-day window.

#### Neural Network Models
We implement several neural network architectures to capture both spatial and temporal dependencies in the data.

- Simple-MLP: A baseline feedforward neural network that captures nonlinear relationships using the same flattened 1,125-dimensional feature vectors as the classical models. This serves as a reference point for what the time-series aware neural models can improve on.
- CNN-LSTM: Combines convolutional layers for local temporal feature extraction with LSTM layers for longer range sequential dependencies. This is our proposed model and we expect it to benefit from both the pattern detection of the CNN and the sequential memory of the LSTM.
- Two-Layer-LSTM: Focuses on sequential dependencies in the data by modeling temporal patterns across multiple periods using two stacked LSTM layers.
- LightTS-Inspired: A lightweight time series model that uses 1D convolutions and global average pooling to efficiently capture temporal structure without the computational overhead of recurrent layers.

These models operate directly on the 3D sequence tensor without flattening and are particularly well-suited for structured spatiotemporal data where both location-based and time-based patterns are important. All neural models use weighted cross-entropy loss to handle class imbalance and are trained with the Adam optimizer.

#### Distance-Based and Probabilistic Models
We also include K-Nearest Neighbors and Naive Bayes as additional benchmarks. KNN captures local similarity in the feature space, while Naive Bayes provides a probabilistic framework under simplifying independence assumptions. These models provide useful comparisons for evaluating more complex approaches.

#### Feature Embeddings with Chronos
After obtaining baseline results, we extracted feature embeddings using a pretrained Chronos T5 Mini model for each of our features. These embeddings are designed to capture richer temporal representations and underlying patterns in the data that may not be captured by raw features alone. From there we re-estimated all models using the new embedded feature set to evaluate whether these representations improve predictive performance across each model.

#### Model Evaluation
All models are evaluated using the same train/test split to ensure consistency in comparison. Predictive performance is measured using accuracy, precision, recall, and F1-score on both the training and test datasets. This allows us to evaluate overall model accuracy while also identifying any potential overfitting. The table reproduced in the results section includes the results for the aforementioned model with and without the embeddings. 

### Modeling Limitations
## Results
## Recommendations
## Reproduction
1. Clone the repository git@github.com:Williamsillon/eco395m-ml-final-project.git
2. Install additional packages pip install -r requirements
