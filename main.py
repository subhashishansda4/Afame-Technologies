# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Data Transformations
# -------------------------------------------------------------------
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

# one-hot encode gender, geography & exited columns
onehot_gen = pd.get_dummies(df['Gender']).astype(int)
onehot_geo = pd.get_dummies(df['Geography']).astype(int)
# remove columns
dfd = df.drop(columns=['RowNumber', 'Surname', 'Geography', 'Gender'])
# merge
df_ = pd.concat([dfd, onehot_gen, onehot_geo], axis=1)




# Exploratory Data Analysis
# -------------------------------------------------------------------
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




# Feature Engineering
# -------------------------------------------------------------------
# product relations
df_['CustomerValue'] = df_.apply(lambda row: row['Tenure'] * row['NumOfProducts'], axis=1)
df_['PurchaseValue'] = df_.apply(lambda row: row['CreditScore'] * row['NumOfProducts'], axis=1)

# ratio relations
df_['Worth'] = df_.apply(lambda row: row['Balance'] / row['Age'] if row['Age'] != 0 else row['Balance'], axis=1)
df_['Affordance'] = df_.apply(lambda row: row['CreditScore'] / row['Tenure'] if row['Tenure'] != 0 else row['CreditScore'], axis=1)

# additive relations
df_['TotalBalance'] = df_.apply(lambda row: row['Balance'] + row['EstimatedSalary'], axis=1)

# subtractive relations
df_['Balance_'] = df_.apply(lambda row: abs(row['Balance'] - row['EstimatedSalary']), axis=1)




# Normalization
# -------------------------------------------------------------------
normalize_cls = df_[['CreditScore', 'Age', 'Balance', 'Tenure', 'NumOfProducts', 'EstimatedSalary', 'CustomerValue', 'PurchaseValue', 'Worth', 'Affordance', 'TotalBalance', 'Balance_']]
for c in normalize_cls.columns:
    df_[c] = (df_[c] - df_[c].min()) / (df_[c].max() - df_[c].min())

temp_df = df_[['Exited']]
final_df = df_.drop(columns=['CustomerId', 'Exited'])
final_df = pd.concat([final_df, temp_df], axis=1)
final_df.to_csv('data/final_df.csv')




# Visualization
# -------------------------------------------------------------------
# t-SNE
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.io as pio
pio.renderers.default = 'browser'

x = final_df
y = pd.DataFrame(final_df, columns=['Exited_0', 'Exited_1']).values

tsne3d = TSNE(
    n_components = 3,
    random_state = 101,
    method = 'barnes_hut',
    n_iter = 1000,
    verbose = 2,
    angle = 0.5
).fit_transform(x)

# 3d plot
tsne3d_one = tsne3d[:,0]
tsne3d_two = tsne3d[:,1]
tsne3d_three = tsne3d[:,2]

px.scatter_3d(
    final_df,
    x = tsne3d_one, y = tsne3d_two, z = tsne3d_three,
    color='Exited_1'
)




# Machine Learning
# -------------------------------------------------------------------
MODEL_SCORES = 'scores.txt'

# 80:20 split
train_percent = 0.8
test_percent = 0.2

# train
train_ml = final_df.sample(int(len(final_df) * train_percent), ignore_index=True)
x_train_ml = train_ml.drop(columns=['Exited'])
y_train_ml = pd.DataFrame(train_ml, columns=['Exited'])
y_train_ml = np.ravel(y_train_ml)

# test
test_ml = final_df.sample(int(len(final_df) * test_percent), ignore_index=True)
x_test_ml = test_ml.drop(columns=['Exited'])
y_test_ml = pd.DataFrame(test_ml, columns=['Exited'])


# cross validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

x_train, x_test, y_train, y_test = train_test_split(train_ml.iloc[:, :-1].values,
                                                    train_ml.iloc[:, -1:].values,
                                                    test_size=0.3, random_state=0)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

kfold = KFold(n_splits=5, shuffle=True, random_state=0)


# models
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

n=5

lr = LogisticRegression()  
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
lda = LinearDiscriminantAnalysis()
knn = KNeighborsClassifier(n)
svc = SVC(probability=True)
nb = GaussianNB()
xgb = XGBClassifier()

models = [lr, dt, rf, lda, knn, svc, nb, xgb]
param_grid = {}
'''
param_grid = {
    "sgdc__loss": [0.1, 1, 10, 100],
    "sgdc__penalty": ['linear', 'poly', 'rbf', 'sigmoid'],
    "sgdc__alpha": [2, 3, 4, 5],
    "sgdc__l1_ratio": ['scale', 'auto'],
    "sgdc__fit_intercept": [True, False],
    "sgdc__max_iter": [1000, 2000, 5000],
    "sgdc__tol": [1e-3, 1e-4, 1e-5],
    
    "dt__criterion": ['gini', 'entropy'],
    "dt__splitter": ['best', 'random'],
    "dt__max_depth": [None, 5, 10, 20],
    "dt__min_samples_split": [2, 5, 10],
    "dt__min_samples_leaf": [1, 2, 4],
    
    "rf__n_estimators": [10, 50, 100, 200],
    "rf__criterion": ['gini', 'entropy'],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__min_weight_fraction_leaf": [0, 0.1, 0.2],
    "rf__max_leaf_nodes": [None, 10, 20, 30],
    "rf__max_depth": [None, 5, 10],
    
    "knn__n_neighbours": [3, 5, 7, 9],
    "knn__weights": ['uniform', 'distance'],
    "knn__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
    
    "svc__C": [0.1, 1, 10, 100],
    "svc__kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "svc__degree": [2, 3, 4, 5],
    "svc__gamma": ['scale', 'auto'],
    
    "xgb__max_depth": [3, 5, 7, 9],
    "xgb__learning_rate": [0.1, 0.2, 0.3],
    "xgb__n_estimators": [100, 200, 300],
    "xgb__gamma": [0, 0.5, 1],
    "xgb__subsample": [0.5, 0.8, 1.0],
    "xgb__colsample_bytree": [0.5, 0.8, 1.0],
    "xgb__reg_alpha": [0, 0.5, 1],
    
    "nb__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
}
'''


# cross validation
s = []
a = []
m = []

def cross_val_train():
    for model in models:
        scrs = []
        accs = []
        
        with open(MODEL_SCORES, 'a', encoding='utf-8') as f:
            f.write('\n' + '\n' + '\n' + '=====================================' + '\n' +
                    '\n' + str(model))
        
        for train, test in kfold.split(x_train):
            grid = GridSearchCV(
                model,
                param_grid,
                cv=kfold,
                scoring='neg_log_loss', refit='neg_log_loss'
            )
            grid.fit(x_train[train], y_train[train])
            
            scr = grid.score(x_train[test], y_train[test])
            scrs.append(scr)
            
            grid = GridSearchCV(
                model,
                param_grid,
                cv=kfold,
                scoring='roc_auc', refit='roc_auc'
            )
            grid.fit(x_train[train], y_train[train])
            
            y_pred = grid.predict(x_train[test])
            acc = metrics.accuracy_score(y_train[test], y_pred)
            accs.append(acc)
            
            mtrx = metrics.confusion_matrix(y_train[test], y_pred)
            
            with open(MODEL_SCORES, 'a', encoding='utf-8') as f:
                f.write('\n' + 'score: {:.2f}'.format(np.mean(scr)) + '\n' +
                        'accuracy: {:.2f} +/- {:.2f}'.format(np.mean(acc), np.std(acc)) + '\n' +
                        '\n' + 'confusion matrix' + '\n' +
                        str(mtrx) + '\n' + '\n' + '--------' + '\n')
            
        s.append(min(scrs))
        a.append(min(accs))
        m.append(model)
        
# kfold
cross_val_train()


# scores table
metrics_df = pd.DataFrame({'Model': m, 'Score': s, 'Accuracy': a})
metrics_df.to_csv('data/metrics_df.csv')


# Chosen Model: Random Forest
grid = GridSearchCV(
    rf,
    param_grid,
    cv=kfold,
)

grid.fit(x_train_ml, y_train_ml)


# Actual vs Predicted
y_pred_ml = pd.DataFrame(grid.predict(x_test_ml), columns=['Predicted'])

out_df = pd.concat([y_test_ml, y_pred_ml], axis=1)
out_df.columns = out_df.columns.astype(str)

# heatmap
plt.figure(figsize=(8,20))
sns.heatmap(out_df)
plt.savefig('plots/Predicted Heatmaps.png', dpi=300)
plt.show()



