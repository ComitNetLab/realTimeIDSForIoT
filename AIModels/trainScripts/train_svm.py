from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.svm import SVC
import pandas as pd
import time
from load_data import load_data, create_pre_processor

attack, x, y = load_data()
x_train, y_train, x_test, y_test, preprocessor = create_pre_processor(x, y, attack)

# Pipeline for transformation and model
estimator = [('clf', SVC(random_state=1234))]
pipeline = Pipeline(estimator)

# Param grid for GridSearch
param_grid = dict(clf__C=[0.001, 0.01, 0.1, 1],
                  clf__kernel=['linear', 'poly', 'rbf', 'sigmoid'],
                  clf__class_weight=['balanced', None],
                  clf__decision_function_shape=['ovo', 'ovr'])

# GridSearch
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=3, scoring='accuracy', error_score='raise')
grid_search.fit(x_train, y_train)

# Save results of GridSearch
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv(f'../trainResults/training-svm-{attack}.csv')

# Save best model
dump(grid_search.best_estimator_, f'../bestModels/svm-{attack}.joblib')

# Load best model
model = load(f'../bestModels/svm-{attack}.joblib')

# Load the preprocessor and transform the test data
loaded_preprocessor = load(f'../trainResults/preprocessor-{attack}.pkl')
p_x_test = preprocessor.transform(x_test).toarray()

# Test metrics
start = time.time()
y_pred = model.predict(p_x_test)
end = time.time()
execution_time = end - start

# Save confusion matrix
cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=[0, 1])
cm.figure_.savefig(f"../testResults/testing-svm-{attack}-image.png")

# Save metrics report
cf = classification_report(y_test, y_pred)
with open(f"../testResults/testing-svm-{attack}-report.txt", 'w') as file:
    file.write(cf)

print('Script Finished!')
