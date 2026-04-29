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

#### Classical Baseline Models
A logistic regression model is used as a baseline benchmark. This provides a simple and interpretable reference point for evaluating more complex models. However, we expect this model to have limitations due to nonlinear relationships and interactions between environmental and weather-related predictors.

Tree-based models such as Decision Trees, Random Forest, Gradient Boosting, and XGBoost are implemented to capture nonlinear relationships and higher-order interactions between predictors. Random Forest helps reduce variance through averaging across trees. Boosting-based methods like Gradient Boosting and XGBoost iteratively improve model performance by focusing on previous errors. These models are particularly useful given that  wildfire risk may depend on complex combinations of weather and environmental conditions. Since these models are not inherently time-series aware, the 3D sequence tensor is flattened to 1,125 features before modeling, meaning they cannot capture the temporal ordering of the 75-day window.

We also include K-Nearest Neighbors and Naive Bayes as additional benchmarks. KNN captures local similarity in the feature space, while Naive Bayes provides a probabilistic framework under simplifying independence assumptions. These models provide useful comparisons for evaluating more complex approaches.

#### Neural Network Models
We implement several neural network architectures to capture both spatial and temporal dependencies in the data.

- Simple-MLP: A baseline feedforward neural network that captures nonlinear relationships using the same flattened 1,125-dimensional feature vectors as the classical models. This serves as a reference point for what the time-series aware neural models can improve on.
- CNN-LSTM (ours): Combines convolutional layers for local temporal feature extraction with LSTM layers for longer range sequential dependencies. This is our proposed model and we expect it to benefit from both the pattern detection of the CNN and the sequential memory of the LSTM.
- Two-Layer-LSTM: Focuses on sequential dependencies in the data by modeling temporal patterns across multiple periods using two stacked LSTM layers.
- LightTS-Inspired: A lightweight time series model that uses 1D convolutions and global average pooling to efficiently capture temporal structure without the computational overhead of recurrent layers.

These models operate directly on the 3D sequence tensor without flattening and are appropriate to use for structured spatiotemporal data where both location-based and time-based patterns matter. All neural models use a weighted cross-entropy loss to handle class imbalance and are trained with the Adam optimizer.

#### Feature Embeddings with Chronos
After obtaining baseline results, we extracted embeddings using a frozen pretrained Amazon Chronos T5 Mini model. Each GRIDMET feature is encoded as a univariate time series, mean-pooled, and concatenated into a flat embedding per sample, then reduced via PCA before classification. The full implementation is in `train/train_table_iii_cloud.py`. We re-ran all baseline classifiers on these embeddings to evaluate whether richer temporal representations improve predictive performance.

#### Model Evaluation
All models are evaluated in dstack using the same train/test split to ensure consistency in comparison. Predictive performance is measured using accuracy, precision, recall, and F1-score on both the training and test datasets. This allows us to evaluate overall model accuracy while also identifying any potential overfitting. The table reproduced in the results section includes the results for the aforementioned model with and without the embeddings. 

*Marco can you elaborate on the use of dstack*
### Modeling Limitations
*Marco??? all I can think of is that the code took a long time. 
## Results
We evaluated all of the aforementioned models using the same train/test split and measured performance across accuracy, precision, recall, F1, ROC-AUC, and PR-AUC. Model training was accelerated using dstack to ensure efficient computation across our models like Gradient Boosting and CNN-LSTM. The results are posted in a [table](output/models_table.txt), and they are also pasted below:
| Model | Accuracy [%] | Precision | Recall | F1 | Threshold | ROC-AUC | PR-AUC | Seconds |
|-------|-------------|-----------|--------|----|-----------|---------|--------|---------|
| CNN-LSTM (ours) | 90.2 | 0.96 | 0.91 | 0.93 | 0.620 | 0.952 | 0.984 | 63.16 |
| XGBoost* | 89.4 | 0.94 | 0.92 | 0.93 | 0.247 | 0.948 | 0.984 | 25.32 |
| Gradient Boosting* | 88.3 | 0.93 | 0.92 | 0.92 | 0.237 | 0.952 | 0.986 | 2022.51 |
| Random Forest* | 86.6 | 0.95 | 0.87 | 0.91 | 0.253 | 0.943 | 0.981 | 126.12 |
| K-Nearest Neighbors* | 65.0 | 0.90 | 0.62 | 0.73 | 0.400 | 0.743 | 0.886 | 15.87 |
| Simple-MLP* | 87.3 | 0.91 | 0.93 | 0.92 | 0.590 | 0.915 | 0.974 | 22.98 |
| Decision Tree* | 76.9 | 0.77 | 1.00 | 0.87 | 0.000 | 0.651 | 0.834 | 184.21 |
| Two-Layer-LSTM | 86.2 | 0.91 | 0.91 | 0.91 | 0.602 | 0.876 | 0.955 | 48.04 |
| LightTS-Inspired | 90.7 | 0.99 | 0.89 | 0.94 | 0.609 | 0.968 | 0.991 | 31.94 |
| Logistic Regression* | 83.4 | 0.87 | 0.92 | 0.90 | 0.433 | 0.876 | 0.961 | 81.32 |
| Naive Bayes* | 83.0 | 0.98 | 0.79 | 0.88 | 0.000 | 0.900 | 0.973 | 0.83 |
| Chronos + XGBoost* | 88.5 | 0.95 | 0.90 | 0.92 | 0.250 | 0.957 | 0.987 | 5.22 |
| Chronos + Gradient Boosting* | 89.2 | 0.95 | 0.90 | 0.93 | 0.234 | 0.961 | 0.988 | 1316.63 |
| Chronos + Random Forest* | 86.6 | 0.98 | 0.84 | 0.91 | 0.260 | 0.957 | 0.986 | 115.76 |
| Chronos + K-Nearest Neighbors* | 71.5 | 0.88 | 0.73 | 0.80 | 0.400 | 0.780 | 0.910 | 4.66 |
| Chronos + Simple-MLP* | 89.2 | 0.96 | 0.90 | 0.93 | 0.579 | 0.944 | 0.980 | 22.19 |
| Chronos + Decision Tree* | 76.9 | 0.77 | 1.00 | 0.87 | 0.000 | 0.616 | 0.816 | 73.62 |
| Chronos + Logistic Regression* | 88.7 | 0.91 | 0.95 | 0.93 | 0.409 | 0.960 | 0.989 | 3.51 |
| Chronos + Naive Bayes* | 81.5 | 0.98 | 0.78 | 0.87 | 0.194 | 0.920 | 0.977 | 0.26 |

These results indicate that our time-series aware models performed competitively against the classical baselines. The LightTS-Inspired model achieved the highest accuracy at 90.7%, an F1 of 0.94, and the highest PR-AUC of 0.991. Our proposed model, the CNN-LSTM, was the next best model with 90.2% accuracy, an F1 of 0.93, and an ROC-AUC of 0.952. This suggests that explicitly modeling the temporal structure of the 75-day sequences provides real predictive value beyond just simply flattening the features.

On the other hand, XGBoost performed the strongest out of the classical baseline models, with 89.4% accuracy. Gradient Boosting also performed well with a 88.3% accuracy, but with a significantly longer training time of 2022 seconds versus XGBoost's 25 seconds. K-Nearest Neighbors was the weakest baseline at 65.0% accuracy. This finding is not surprising given that distance-based methods tend to scale poorly when dealing with high-dimensional flattened feature vectors.

Regarding the Chronos embedding results, we find that the embedding resulted in modest, but positive improvements across most models, except for the XGBoost and Naive Bayes models. Logistic Regression saw a large improvement, going from 83.4% accurarcy without the embeddings to 88.7% accuracy with the embeddings. This suggests these pretrained temporal representations captured structure that the linear Logistic regression model cannot learn on its own. Additionally, the Chronos embeddings also improved KNN from 65.0% accurary to 71.5% accurary.
## Recommendations
*????
we recommend you hire Always Ready Analytics*
## Reproduction
1. Clone the repository `git@github.com:Williamsillon/eco395m-ml-final-project.git`
2. Install additional packages `pip install -r requirements`
