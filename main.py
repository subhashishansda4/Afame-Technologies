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





sns.set(style='whitegrid')
numerical_cls = df_[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']]
categorical_cls = df[['Gender', 'Geography', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']]

r = 3
c = 2

# numerical feature distributions
# hist plots for Exited
fig, axes = plt.subplots(r, c, figsize=(20, 12))
for i, (col, ax) in enumerate(zip(numerical_cls.columns, axes.flat)):
    hist = sns.histplot(data=df, x=col, hue='Exited', ax=ax, multiple='stack', palette='Pastel1')
plt.tight_layout()
plt.savefig('plots/Exited-numerical.png', dpi=300)
plt.show()

# hist plots for Gender
fig, axes = plt.subplots(r, c, figsize=(20, 12))
for i, (col, ax) in enumerate(zip(numerical_cls.columns, axes.flat)):
    hist = sns.histplot(data=df, x=col, hue='Gender', ax=ax, multiple='stack', palette='Pastel1')
plt.tight_layout()
plt.savefig('plots/Gender-numerical.png', dpi=300)
plt.show()

# hist plots for Geography
fig, axes = plt.subplots(r, c, figsize=(20, 12))
for i, (col, ax) in enumerate(zip(numerical_cls.columns, axes.flat)):
    hist = sns.histplot(data=df, x=col, hue='Geography', ax=ax, multiple='stack', palette='Pastel1')
plt.tight_layout()
plt.savefig('plots/Geography-numerical.png', dpi=300)
plt.show()

# hist plots for NumOfProducts
fig, axes = plt.subplots(r, c, figsize=(20, 12))
for i, (col, ax) in enumerate(zip(numerical_cls.columns, axes.flat)):
    hist = sns.histplot(data=df, x=col, hue='NumOfProducts', ax=ax, multiple='stack', palette='Pastel1')
plt.tight_layout()
plt.savefig('plots/NumOfProducts-numerical.png', dpi=300)
plt.show()

# hist plots for HasCrCard
fig, axes = plt.subplots(r, c, figsize=(20, 12))
for i, (col, ax) in enumerate(zip(numerical_cls.columns, axes.flat)):
    hist = sns.histplot(data=df, x=col, hue='HasCrCard', ax=ax, multiple='stack', palette='Pastel1')
plt.tight_layout()
plt.savefig('plots/HasCrCard-numerical.png', dpi=300)
plt.show()

# hist plots for IsActiveMember
fig, axes = plt.subplots(r, c, figsize=(20, 12))
for i, (col, ax) in enumerate(zip(numerical_cls.columns, axes.flat)):
    hist = sns.histplot(data=df, x=col, hue='IsActiveMember', ax=ax, multiple='stack', palette='Pastel1')
plt.tight_layout()
plt.savefig('plots/IsActiveMember-numerical.png', dpi=300)
plt.show()





# categorical variable distributions
# count plots for Exited
fig, axes = plt.subplots(r, c, figsize=(20, 16))
for i, (col, ax) in enumerate(zip(categorical_cls.columns, axes.flat)):
    count = sns.countplot(data=df, x=df['Exited'], hue=col, ax=ax, palette='Pastel2')
    for p in count.patches:
        height = p.get_height()
        if height != 0:
            count.annotate(f'{int(height)}', (p.get_x() + p.get_width()/2., height), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('plots/Exited-categorical.png', dpi=300)
plt.show()

# count plots for Gender
fig, axes = plt.subplots(r, c, figsize=(20, 16))
for i, (col, ax) in enumerate(zip(categorical_cls.columns, axes.flat)):
    count = sns.countplot(data=df, x=df['Gender'], hue=col, ax=ax, palette='Pastel2')
    for p in count.patches:
        height = p.get_height()
        if height != 0:
            count.annotate(f'{int(height)}', (p.get_x() + p.get_width()/2., height), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('plots/Gender-categorical.png', dpi=300)
plt.show()

# count plots for Geography
fig, axes = plt.subplots(r, c, figsize=(20, 16))
for i, (col, ax) in enumerate(zip(categorical_cls.columns, axes.flat)):
    count = sns.countplot(data=df, x=df['Geography'], hue=col, ax=ax, palette='Pastel2')
    for p in count.patches:
        height = p.get_height()
        if height != 0:
            count.annotate(f'{int(height)}', (p.get_x() + p.get_width()/2., height), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('plots/Geography-categorical.png', dpi=300)
plt.show()

# count plots for NumOfProducts
fig, axes = plt.subplots(r, c, figsize=(20, 16))
for i, (col, ax) in enumerate(zip(categorical_cls.columns, axes.flat)):
    count = sns.countplot(data=df, x=df['NumOfProducts'], hue=col, ax=ax, palette='Pastel2')
    for p in count.patches:
        height = p.get_height()
        if height != 0:
            count.annotate(f'{int(height)}', (p.get_x() + p.get_width()/2., height), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('plots/NumOfProducts-categorical.png', dpi=300)
plt.show()

# count plots for HasCrCard
fig, axes = plt.subplots(r, c, figsize=(20, 16))
for i, (col, ax) in enumerate(zip(categorical_cls.columns, axes.flat)):
    count = sns.countplot(data=df, x=df['HasCrCard'], hue=col, ax=ax, palette='Pastel2')
    for p in count.patches:
        height = p.get_height()
        if height != 0:
            count.annotate(f'{int(height)}', (p.get_x() + p.get_width()/2., height), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('plots/HasCrCard-categorical.png', dpi=300)
plt.show()

# count plots for IsActiveMember
fig, axes = plt.subplots(r, c, figsize=(20, 16))
for i, (col, ax) in enumerate(zip(categorical_cls.columns, axes.flat)):
    count = sns.countplot(data=df, x=df['IsActiveMember'], hue=col, ax=ax, palette='Pastel2')
    for p in count.patches:
        height = p.get_height()
        if height != 0:
            count.annotate(f'{int(height)}', (p.get_x() + p.get_width()/2., height), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('plots/IsActiveMember-categorical.png', dpi=300)
plt.show()




