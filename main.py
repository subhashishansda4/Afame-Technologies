# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# load data
df = pd.read_csv('data/Churn_Modelling.csv')
df.info()

# missing and duplicate values
print('')
print('--------- Missing Values ---------')
print(df.isnull().sum())

print('')
print('--------- Duplicate Rows ---------')
print(df.duplicated().sum())

# one-hot encode gender & geography columns
onehot_gen = pd.get_dummies(df['Gender']).astype(int)
onehot_geo = pd.get_dummies(df['Geography']).astype(int)
# remove columns
dfd = df.drop(columns=['RowNumber', 'Surname', 'Geography', 'Gender'])
# merge
df_ = pd.concat([dfd, onehot_gen, onehot_geo], axis=1)


# numerical feature distributions
sns.set(style='whitegrid')

# hist plots b/n numerical features & exited
numerical_cls = df_[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']]
plt.figure()
hist = sns.FacetGrid(df_, col='Exited', height=4, aspect=1)
for c in numerical_cls.columns:
    hist.map(sns.histplot, c)
plt.tight_layout()
plt.savefig('plots/' + c)
plt.show()
        
# count plots b/n categorical features & exited
categorical_cls = df[['Gender', 'Geography', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']]
for c in categorical_cls.columns:
    plt.figure()
    ax = sns.countplot(x=df['Exited'], hue=c, data=df, palette='Pastel1')
    plt.xlabel('Exited')
    plt.ylabel(c)
    for p in ax.patches:
        height = p.get_height()
        if height != 0:
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width()/2., height), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('plots/' + c)
    plt.show()

# hist plots b/n numerical and categorical variables

# Create a grid of subplots
fig, axes = plt.subplots(nrows=1, ncols=len(numerical_cls), figsize=(15, 5))

# Create scatter plots for each numerical feature
for i, feature in enumerate(numerical_cls):
    # Create scatter plot
    sns.scatterplot(x=feature, y='Exited', data=df_, ax=axes[i])
    
    # Add labels and title
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Exited')
    axes[i].set_title(f'Scatter Plot between {feature} and Exited')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
