## Introduction
Developed a machine learning model to predict **Customer Churn** for a subscription-based service or business. I have used historical customer data, including features like usage behaviour and customer demographics

## Raw Data
### Customer Data
![1](https://github.com/subhashishansda4/Afame-Technologies/blob/main/data/df_.png)

### Data Informatics
![2](https://github.com/subhashishansda4/Afame-Technologies/blob/main/data/df_info.png)

## Exploratory Data Analysis
### Numerical Feature Distributions
Exited\
![3](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/Exited-numerical.png)

Gender\
![4](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/Gender-numerical.png)

Geography\
![5](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/Geography-numerical.png)

HasCrCard\
![6](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/HasCrCard-numerical.png)

IsActiveMember\
![7](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/IsActiveMember-numerical.png)

NumOfProducts\
![8](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/NumOfProducts-numerical.png)

### Categorical Variables
Exited\
![9](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/Exited-categorical.png)

Gender\
![10](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/Gender-categorical.png)

Geography\
![11](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/Geography-categorical.png)

HasCrCard\
![12](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/HasCrCard-categorical.png)

IsActiveMember\
![13](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/IsActiveMember-categorical.png)

NumOfProducts\
![14](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/NumOfProducts-categorical.png)

## Feature Engineering
### Product Relations
* **Customer Value** = Tenure * NumOfProducts
* **Purchase Value** = CreditScore * NumOfProducts

### Ratio Relations
* **Worth** = Balance / Age
* **Affordance** = CreditScore / Tenure

### Additive Relations
* **Total Balance** = Balance + EstimatedSalary

### Subtractive Relations
* **Balance_** = Balance - EstimatedSalary

## Normalization
### Final Dataset
![15](https://github.com/subhashishansda4/Afame-Technologies/blob/main/data/final_df_.png)

### t-SNE
https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/t-SNE.mp4

## Model Evaluation
Used **KFold** for model evaluation from 8 different classification models\
Used **LogLoss**, **Accuracy** under **ROC curve** and **Confusion Matrix** as scoring parameters

Selected **Random Forest Classifier** as the most suitable model

![17](https://github.com/subhashishansda4/Afame-Technologies/blob/main/data/metrics_df_.png)

## Predictions
![18](https://github.com/subhashishansda4/Afame-Technologies/blob/main/plots/Predicted-Heatmaps.png)








