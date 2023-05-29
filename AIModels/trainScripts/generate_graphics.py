from sklearn.metrics import classification_report
from joblib import load
import time
from trainScripts.load_data import load_data, create_pre_processor

attack, data, x, y = load_data()
x_test, y_test = create_pre_processor(x, y, attack)

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
