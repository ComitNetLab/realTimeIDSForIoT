from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from joblib import load
import pandas as pd
import time



enc = OrdinalEncoder(encoded_missing_value=-1)
x_trans = enc.fit_transform(x)

# Separate train and test
x_train, x_test, y_train, y_test = train_test_split(x_trans, y, test_size=0.2, random_state=123, shuffle=True)

# Load best model
model = load(f'../bestModels/svm-{attack}.joblib')

# Test metrics
start = time.time()
y_pred = model.predict(x_test)
end = time.time()
execution_time = end - start

# Save confusion matrix
cm = plot_confusion_matrix(model, x_test, y_test, display_labels=[0, 1], values_format='d')
cm.figure_.savefig(f"../testResults/testing-svm-{attack}-image.png")

# Save metrics report
cf = classification_report(y_test, y_pred)
with open(f"../testResults/testing-svm-{attack}-report.txt", 'w') as file:
    file.write(cf)
