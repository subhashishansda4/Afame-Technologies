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
df_.hist(figsize=(15,15))
plt.subplots_adjust(bottom=0.1)
plt.title('Numerical Feature Distributions')
plt.show()

# scatter plots b/n numerical features & exited
numerical_cls = df_.select_dtypes(include=[np.number])
for c in numerical_cls.columns:
    if c != 'CustomerId':
        sns.scatterplot(x=df_['Exited'], y=df_[c])
        plt.xlabel(c)
        plt.ylabel('Exited')
        plt.title(c + ' vs ' + 'Exited')
        plt.show()
        
# box plots b/n categorical features & exited
categorical_cls = df[['Gender', 'Geography']]
for c in categorical_cls.columns:
    sns.boxplot(x=df['Exited'], y=c, data=df)
    plt.xlabel('Exited')
    plt.ylabel(c)
    plt.title('Exited' + ' vs ' + c)
    plt.show()


