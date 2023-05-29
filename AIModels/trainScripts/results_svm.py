from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.svm import SVC
import pandas as pd
import time
from load_data import load_data, create_pre_processor


attack, x, y = load_data()
x_train, y_train, x_test, y_test, preprocessor = create_pre_processor(x, y, attack)

# Load best model
model = load(f'../bestModels/svm-{attack}.joblib')

# Load the preprocessor and transform the test data
loaded_preprocessor = load(f'../trainResults/preprocessor-{attack}.pkl')
p_x_test = preprocessor.transform(x_test)#.toarray()

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
