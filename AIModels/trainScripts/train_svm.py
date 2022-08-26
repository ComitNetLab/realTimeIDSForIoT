from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import time
import sys

# Constants
DATASETS = {'ddos': '../Traning and Testing Tets (5% of the entier dataset)/All '
                    'features/UNSW_2018_IoT_Botnet_Full5pc_1.csv',
            'dos': 'files',
            'scan': 'files',
            'theft': 'files'}

# Get the features and category of attack selected for training
if len(sys.argv) < 3:
    raise ValueError('Please provide the features and attack to train.')

attack = sys.argv[1].lower()
features = [f.lower() for f in sys.argv[2:]]

# Load the dataset
data = pd.read_csv(DATASETS[attack], low_memory=False)

# Change columns names to lower case
data.columns = data.columns.str.lower()

# Get features and labels
x = data[features]
y = data['attack']

# Separate train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

# Pipeline for transformation and model
ct = ColumnTransformer([
    ('num', SimpleImputer(strategy='mean'),
     make_column_selector(dtype_include=np.number)),
    ('cat', SimpleImputer(strategy='most_frequent'),
     make_column_selector(dtype_include=object))])

estimators = [('imputer', ct),
              ('normalize', MinMaxScaler(feature_range=(0, 1))),
              ('clf', SVC(random_state=1234, class_weight='balanced'))]
pipeline = Pipeline(estimators)
pipeline

# Param grid for GridSearch
param_grid = dict(imputer__num__strategy=['mean', 'median'],
                  imputer__cat__strategy=['most_frequent'],
                  normalize=[MinMaxScaler(feature_range=(0, 1))],
                  clf__C=[0, 0.01, 0.1, 1],
                  clf__kernel=['linear', 'poly', 'rbf', 'sigmoid'],
                  clf__class_weight=['balanced', None],
                  clf__decision_function_shape=['ovo', 'ovr'])

# GridSearch
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=3, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Save results of GridSearch
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv(f'../Results/training-svm-{attack}.csv')

# Save best model
dump(grid_search.best_estimator_, f'../BestModels/svm-{attack}.joblib')

# Load best model
model = load(f'../BestModels/svm-{attack}.joblib')

# Test metrics
start = time.time()
y_pred = model.predict(x_test)
end = time.time()
execution_time = end-start

# Save confusion matrix
cm = plot_confusion_matrix(model, x_test, y_test, display_labels=[0, 1], values_format='d')
cm.figure_.savefig(f"../Results/testing-svm-{attack}.png")

# Save metrics report
cf = classification_report(y_test, y_pred)
df_cf = pd.DataFrame(cf)
df_cf.to_csv(f"../Results/testing-svm-{attack}.csv")
