from sklearn.metrics import classification_report, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from tensorflow.keras.layers import Dense, Dropout, Input
# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.keras.models import Sequential
import tensorflow as tf

from joblib import dump, load
import pandas as pd
import numpy as np
import time
import sys

# Constants
PATH = 'C:/Users/server22/Downloads/tesisInfo/data/grouped/'
DATASET = {'normal': PATH + 'IoT_Botnet_Dataset_Normal_Traffic.csv',
           'ddos': PATH + 'IoT_Botnet_dataset_DDoS_FULL_Traffic.csv',
           'dos': PATH + 'IoT_Botnet_Dataset_DoS_FULL_Traffic.csv',
           'os_fingerprint': PATH + 'IoT_Botnet_Dataset_OS_Fingerprint_FULL_Traffic.csv',
           'service_scan': PATH + 'IoT_Botnet_Dataset_Service_Scan_FULL_Traffic.csv',

           'keylogging': PATH + 'IoT_Botnet_Dataset_Keylogging_FULL_Traffic.csv',
           'keylogging_normal': PATH + 'IoT_Botnet_Dataset_Normal_Keylogging_Traffic.csv',
           'data_exfiltration': PATH + 'IoT_Botnet_Dataset_Data_Exfiltration_FULL_Traffic.csv',
           'data_exfiltration_normal': PATH + 'IoT_Botnet_Dataset_Normal_Data_Exfiltration_Traffic.csv',
           }

# Get the features and category of attack selected for training
if len(sys.argv) < 3:
    raise ValueError('Please provide the features and attack to train.')

attack = sys.argv[1].lower()
features = [f.lower() for f in sys.argv[2:]]

# Load the dataset
all_files = [DATASET[attack], DATASET[attack + '_normal']]
cols = ['pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport', 'pkts', 'bytes', 'state', 'ltime',
        'seq', 'dur', 'mean', 'stddev', 'smac', 'dmac', 'sum', 'min', 'max', 'soui', 'doui', 'sco', 'dco', 'spkts',
        'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate', 'attack', 'category', 'subcategory']
# 'flgs' 'proto' 'sport' 'dport' 'pkts' 'bytes' 'state' 'ltime' 'dur' 'mean' 'stddev' 'smac' 'dmac' 'sum' 'min' 'max'
# 'soui' 'doui' 'sco' 'dco' 'spkts' 'dpkts' 'sbytes' 'dbytes' 'rate' 'srate' 'drate' 'attack'
data = pd.concat((pd.read_csv(f, low_memory=False, names=cols) for f in all_files), ignore_index=True)

# Change columns names to lower case
data.columns = data.columns.str.lower()

# Change type of ports columns to avoid errors caused by null values
# TODO Mari estuvo aquí, así se cambia a float
# data = data.replace('0x0303', np.NaN).replace('0xa549', np.NaN).replace('0x80d3', np.NaN).replace('0x72ba', np.NaN)
data.sport = data.sport.astype(str)
data.dport = data.dport.astype(str)

# Get features and labels
x = data[features]
y = data['attack']

# Separate train and test
x_train, x_test, y_train, y_test = train_test_split(x_trans, y, test_size=0.2, random_state=123, shuffle=True)

# Transformers for different datatypes
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean', fill_value=-1)),
    ('scaler', MinMaxScaler(feature_range=(0, 1)))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('encoder', OneHotEncoder())
])

# Arrays with features names for each datatype
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Join transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ]
)

# Train de preprocessor in all the data & transform the train data
preprocessor = preprocessor.fit(x)
x_train = preprocessor.transform(x_train).toarray()

# Save preprocessor pipeline
dump(preprocessor, f'./trainResults/preprocessor-rn-{attack}.pkl')


# Define Neural Network
def train_nn(nn1=512, nn2=128, dropout=0.5, lr=0.001, hidden_activation='relu'):
    nn = Sequential(name='nn_attacks')
    output = 1
    # Input layer
    nn.add(Input(batch_input_shape=(None, len(features))))
    # First hidden layer
    nn.add(Dense(nn1, activation=hidden_activation))
    # Second hidden layer
    nn.add(Dense(nn2, activation=hidden_activation))
    # Dropout rate
    nn.add(Dropout(dropout, name='Dropout_{0}'.format(dropout)))
    # Ouput layer, softmax activiation
    nn.add(Dense(output, activation='softmax', name='Capa_Salida'))

    nn.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
               metrics=['accuracy'])

    return nn


# Model used by GridSearch
# TODO epochs can change
modelCV = KerasClassifier(model=train_nn, verbose=1, validation_split=0.2, shuffle=True)

# Pipeline for transformation and model
estimators = [('clf', modelCV)]
pipeline = Pipeline(estimators)

# Param grid for GridSearch
# TODO Cambiar valores de nn1 y nn2 por potencias de 2 menores que el número de features y mayores a 1
param_grid = dict(clf__model__nn1=[16, 8],
                  clf__model__nn2=[4, 2],
                  clf__model__dropout=[0.0, 0.1, 0.2, 0.5],
                  clf__model__hidden_activation=['relu', 'sigmoid'],
                  clf__batch_size=[40, 80, 160],
                  clf__epochs=[10, 20, 50]
                  )

# GridSearch
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=3, scoring='accuracy',
                           error_score='raise')
grid_search.fit(x_train, y_train)

# Save results of GridSearch
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv(f'../trainResults/training-rn-{attack}.csv')

# Save rn weights
grid_search.best_estimator_['clf'].model_.save(f'../bestModels/rn-{attack}.h5')
# Save model training history
dump(grid_search.best_estimator_['clf'].history_, f'./trainResults/history-rn-{attack}.pkl')


# Load best model
loaded_model = tf.keras.models.load_model(f'../bestModels/rn-{attack}.h5')

# Load the preprocessor and transform the test data
loaded_preprocessor = load(f'./trainResults/preprocessor-rn-{attack}.pkl')
p_x_test = preprocessor.transform(x_test).toarray()

# Test metrics
start = time.time()
y_pred = loaded_model.predict(p_x_test)
end = time.time()
execution_time = end - start
y_pred = [y[0] for y in y_pred]

# Save confusion matrix
cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=[0, 1])
cm.figure_.savefig(f"../testResults/testing-rn-{attack}-image.png")

# Save metrics report
cf = classification_report(y_test, y_pred)
with open(f"../testResults/testing-rn-{attack}-report.txt", 'w') as file:
    file.write(cf)
