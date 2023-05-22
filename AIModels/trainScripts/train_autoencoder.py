from sklearn.metrics import classification_report, ConfusionMatrixDisplay, \
    recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler, \
    OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from tensorflow.keras.layers import Dense, Dropout, Input
# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.keras.models import Sequential
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

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
all_files = [DATASET[attack], DATASET['normal']]
cols = ['pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport', 'pkts', 'bytes', 'state', 'ltime',
        'seq', 'dur', 'mean', 'stddev', 'smac', 'dmac', 'sum', 'min', 'max', 'soui', 'doui', 'sco', 'dco', 'spkts',
        'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate', 'attack', 'category', 'subcategory']
# 'flgs' 'proto' 'sport' 'dport' 'pkts' 'bytes' 'state' 'ltime' 'dur' 'mean' 'stddev' 'smac' 'dmac' 'sum' 'min' 'max'
# 'soui' 'doui' 'sco' 'dco' 'spkts' 'dpkts' 'sbytes' 'dbytes' 'rate' 'srate' 'drate' 'attack'
data = pd.concat((pd.read_csv(f, low_memory=False, names=cols) for f in all_files), ignore_index=True)

# Change columns names to lower case
data.columns = data.columns.str.lower()

# Change type of ports columns to avoid errors caused by null values
# data = data.replace('0x0303', np.NaN).replace('0xa549', np.NaN).replace('0x80d3', np.NaN).replace('0x72ba', np.NaN)
data.sport = data.sport.astype(str)
data.dport = data.dport.astype(str)

# Get features and labels
x = data[features]
y = data['attack']

enc = OrdinalEncoder(encoded_missing_value=-1)
x_trans = enc.fit_transform(x)

# Separate train and test
x_train, x_test, y_train, y_test = train_test_split(x_trans, y, test_size=0.2, random_state=123, shuffle=True)

# Transformers for different datatypes
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean', fill_value=-1))
    , ('scaler', MinMaxScaler(feature_range=(0, 1)))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant'))
    , ('encoder', OneHotEncoder())
])

# Arrays with features names for each datatype
#
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
x_train.shape

# Save preprocessor pipeline
dump(preprocessor, f'./trainResults/preprocessor-autoencoder-{attack}.pkl')


# Define Neural Network
def train_nn(nn1=512, nn2=128, dropout=0.5, lr=0.001, hidden_activation='relu'):
    nn = Sequential(name='nn_attacks')
    # Input layer
    nn.add(Input(batch_input_shape=(None, x_train.shape[1]), name='Capa_Entrada'))
    # Encoder
    nn.add(Dense(nn1, activation=hidden_activation, name='Encoder'))
    nn.add(Dropout(dropout, name='Dropout_1_{0}'.format(dropout)))
    # Bottleneck
    nn.add(Dense(nn2, activation=hidden_activation, name='Bottleneck'))
    # Decoder
    nn.add(Dense(nn1, activation=hidden_activation, name='Decoder'))
    nn.add(Dropout(dropout, name='Dropout_2_{0}'.format(dropout)))
    # Ouput layer, softmax activiation
    nn.add(Dense(x_train.shape[1], activation='tanh', name='Capa_Salida'))

    nn.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
               metrics=['mse', 'mae'])

    print(nn.summary())
    return nn


# Model used by GridSearch
modelCV = KerasRegressor(model=train_nn, verbose=1, validation_split=0.2, shuffle=True)

# Pipeline for model
estimator = [('autoencoder', modelCV)]
pipeline = Pipeline(estimator)

# Param grid for GridSearch
# TODO Cambiar valores de nn1 y nn2 por potencias de 2 menores que el número de features y mayores a 1
param_grid = dict(autoencoder__model__nn1=[16, 8],
                  autoencoder__model__nn2=[4, 2],
                  autoencoder__model__dropout=[0.0, 0.1, 0.2, 0.5],
                  autoencoder__model__hidden_activation=['relu' 'sigmoid'],
                  autoencoder__batch_size=[40, 80, 160],
                  autoencoder__epochs=[10, 20, 50]
                  )

# GridSearch
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=1, error_score='raise')
# TODO Obtener como parámetro la clase con la que se quiere entrenar el autoencoder
grid_search.fit(x_train[y_train == 0], x_train[y_train == 0])

# Save results of GridSearch
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv(f'../trainResults/training-autoencoder-{attack}.csv')

# Save best autoencoder weights
grid_search.best_estimator_['autoencoder'].model_.save(f'../bestModels/autoencoder-{attack}.h5')
# Save model training history
dump(grid_search.best_estimator_['autoencoder'].history_, f'./trainResults/history-autoencoder-{attack}.pkl')

# Load best model
loaded_model = tf.keras.models.load_model(f'../bestModels/autoencoder-{attack}.h5')


# Save the model architecture
def print_summary(s):
    with open(f'./model-summary-autoencoder-{attack}.txt', 'a') as f:
        print(s, file=f)


loaded_model.summary(print_fn=print_summary)

# Load the preprocessor and transform the test data
loaded_preprocessor = load(f'./trainResults/preprocessor-autoencoder-{attack}.pkl')

p_x_test = preprocessor.transform(x_test).toarray()

# Test metrics
start = time.time()
x_pred_normal = loaded_model.predict(p_x_test[y_test == 0])
x_pred_attack = loaded_model.predict(p_x_test[y_test == 1])
end = time.time()
execution_time = end - start

# Calculate MSE
normal_loss = tf.keras.losses.mse(
    x_pred_normal, p_x_test[y_test == 0]
)

attack_loss = tf.keras.losses.mse(
    x_pred_attack, p_x_test[y_test == 1]
)

normal_error = pd.DataFrame({'reconstruction_error': normal_loss,
                             'true_class': y_test[y_test == 0]})
attack_error = pd.DataFrame({'reconstruction_error': attack_loss,
                             'true_class': y_test[y_test == 1]})
error_df = pd.concat([normal_error, attack_error])
error_df.to_csv(f'../testResults/error-autoencoder-{attack}.csv')

# Visualize error
threshold_fixed = np.mean(normal_loss) + np.std(normal_loss)
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label="Attack" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for normal and attcak data")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.savefig(f"../testResults/error-autoencoder-{attack}-image.png")

# Evaluate performance from anomaly detection
threshold_fixed = np.mean(normal_loss) + np.std(normal_loss)
y_pred = [1 if e > threshold_fixed else 0 for e in error_df.reconstruction_error.values]
error_df['pred'] = y_pred

# Save confusion matrix
cm = ConfusionMatrixDisplay.from_predictions(error_df.true_class, y_pred, labels=[0, 1])
cm.figure_.savefig(f"../testResults/testing-autoencoder-{attack}-image.png")

# Save metrics report
cf = classification_report(error_df.true_class, y_pred)
with open(f"../testResults/testing-autoencoder-{attack}-report.txt", 'w') as file:
    file.write(cf)
