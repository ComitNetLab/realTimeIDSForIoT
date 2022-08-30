from _distutils_hack import shim
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import time
import glob
import os
import sys

# Constants
PATH = '/home/andres/data/cDataSet/'
DATASET = {'normal': PATH + 'IoT_Botnet_Dataset_Normal_Traffic.csv',
           'ddos': PATH + 'UNSW_2018_IoT_Botnet_Dataset_1.csv',
           'dos': PATH + 'path',
           'theft': PATH + 'path',
           'scan': PATH + 'path'}

# Get the features and category of attack selected for training
if len(sys.argv) < 3:
    raise ValueError('Please provide the features and attack to train.')

attack = sys.argv[1].lower()
features = [f.lower() for f in sys.argv[2:]]

# Load the dataset
all_files = [DATASET[attack], DATASET['normal']]
cols = ['pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport', 'pkts', 'bytes', 'state', 'ltime',
           'seq', 'dur', 'mean', 'stddev', 'smac', 'dmac', 'sum', 'min', 'max', 'soui', 'doui', 'sco', 'dco', 'spkts',
           'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate', 'attack', 'category', 'subcategory']
data = pd.concat((pd.read_csv(f, low_memory=False, names=cols) for f in all_files), ignore_index=True)

# Change columns names to lower case
data.columns = data.columns.str.lower()

# Get features and labels
x = data[features]
y = data['attack']

# Separate train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123, shuffle=True)

# Pipeline for transformation and model
estimators = [('encoder', OrdinalEncoder(encoded_missing_value=-1)),
              ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
              ('normalize', MinMaxScaler(feature_range=(0, 1))),
              ('clf', SVC(random_state=1234))]
pipeline = Pipeline(estimators)

# Param grid for GridSearch
param_grid = dict(clf__C=[0, 0.01, 0.1, 1],
                  clf__kernel=['linear', 'poly', 'rbf', 'sigmoid'],
                  clf__class_weight=['balanced', None],
                  clf__decision_function_shape=['ovo', 'ovr'])

# GridSearch
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=3, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Save results of GridSearch
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv(f'../trainResults/training-svm-{attack}.csv')

# Save best model
dump(grid_search.best_estimator_, f'../bestModels/svm-{attack}.joblib')

# Load best model
model = load(f'../bestModels/svm-{attack}.joblib')

# Test metrics
start = time.time()
y_pred = model.predict(x_test)
end = time.time()
execution_time = end - start

# Save confusion matrix
cm = plot_confusion_matrix(model, x_test, y_test, display_labels=[0, 1], values_format='d')
cm.figure_.savefig(f"../testResults/testing-svm-{attack}.png")

# Save metrics report
cf = classification_report(y_test, y_pred)
df_cf = pd.DataFrame(cf)
df_cf.to_csv(f"../testResults/testing-svm-{attack}.csv")
