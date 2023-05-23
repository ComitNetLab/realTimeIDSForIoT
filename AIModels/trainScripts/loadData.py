from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
import sys


def loadData():
    print(sys.argv)
    # Verify the data folders
    if len(sys.argv) < 3:
        raise ValueError('Please provide the data files path.')
    PATH = sys.argv[1]

    # Constants
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
    if len(sys.argv) < 4:
        raise ValueError('Please provide the features and attack to train.')

    attack = sys.argv[2].lower()
    features = [f.lower() for f in sys.argv[3:]]

    # Load the dataset
    all_files = [DATASET[attack], DATASET['normal']]
    cols = ['pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport', 'pkts', 'bytes', 'state', 'ltime',
            'seq', 'dur', 'mean', 'stddev', 'smac', 'dmac', 'sum', 'min', 'max', 'soui', 'doui', 'sco', 'dco', 'spkts',
            'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate', 'attack', 'category', 'subcategory']
    # 'flgs' 'proto' 'sport' 'dport' 'pkts' 'bytes' 'state' 'ltime' 'dur' 'mean' 'stddev' 'smac' 'dmac' 'sum' 'min'
    # 'max' 'soui' 'doui' 'sco' 'dco' 'spkts' 'dpkts' 'sbytes' 'dbytes' 'rate' 'srate' 'drate' 'attack'
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

    return PATH, DATASET, attack, features, all_files, cols, data, x, y


def createPreProcessor(x, y, attack):
    # Separate train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123, shuffle=True)

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
    numeric_features = x.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = x.select_dtypes(include=['object']).columns

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
    dump(preprocessor, f'../trainResults/preprocessor-rn-{attack}.pkl')

    return x_train, y_train, x_test, y_test, preprocessor

