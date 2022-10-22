from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.svm import SVC
import pandas as pd
import time
import sys

# Constants
PATH = '/home/andres/data/grouped/'
DATASET = {'normal': PATH + 'IoT_Botnet_Dataset_Normal_Traffic.csv',
           'ddos': PATH + 'IoT_Botnet_dataset_DDoS_FULL_Traffic.csv',
           'dos': PATH + 'IoT_Botnet_Dataset_DoS_FULL_Traffic.csv',
           'os_fingerprint': PATH + 'IoT_Botnet_Dataset_OS_Fingerprint_FULL_Traffic.csv',
           'service_scan': PATH + 'IoT_Botnet_Dataset_Service_Scan_FULL_Traffic.csv',
           'keylogging': PATH + 'IoT_Botnet_Dataset_Keylogging_FULL_Traffic.csv'}

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
# 'flgs' 'proto' 'sport' 'dport' 'pkts' 'bytes' 'state' 'ltime' 'dur' 'mean' 'stddev' 'smac' 'dmac' 'sum' 'min' 'max'
# 'soui' 'doui' 'sco' 'dco' 'spkts' 'dpkts' 'sbytes' 'dbytes' 'rate' 'srate' 'drate' 'attack'
data = pd.concat((pd.read_csv(f, low_memory=False, names=cols) for f in all_files), ignore_index=True)

# Change columns names to lower case
data.columns = data.columns.str.lower()

# Get features and labels
x = data[features]
y = data['attack']

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
