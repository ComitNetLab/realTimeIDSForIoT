from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
import sys


def check_port_data(x: str) -> bool:
    """
    Check the anomalies values in the ports fields. The 0x are ports that are mapped to hex values in a argus,
    and two cases "login" and "nut" that are on the os_fingerprint and service_scan data
    """
    return str(x).__contains__('x') or str(x).__contains__('login') or str(x).__contains__('nut')


def load_data():
    """
    Function that loads the botIoT data grouped by attacks that are going to be used for algorithms training.
    The function returns the attack name for the preprocessor saving creation,
    in x, the features to be used in the training
    in y, the objective variable of the training
    The data is preprocessed to exclude the columns that are not necessary from the pre analysis given by argus
    open source tool and to transform the data to the needed types.
    :return: attack name, x and y values for training
    """
    # Verify the data folders
    if len(sys.argv) < 3:
        raise ValueError('Please provide the data files path.')
    path = sys.argv[1]

    # Constants
    dataset = {'normal': path + 'IoT_Botnet_Dataset_Normal_Traffic.csv',
               'ddos': path + 'IoT_Botnet_dataset_DDoS_FULL_Traffic.csv',
               'dos': path + 'IoT_Botnet_Dataset_DoS_FULL_Traffic.csv',
               'os_fingerprint': path + 'IoT_Botnet_Dataset_OS_Fingerprint_FULL_Traffic.csv',
               'service_scan': path + 'IoT_Botnet_Dataset_Service_Scan_FULL_Traffic.csv',

               'keylogging': path + 'IoT_Botnet_Dataset_Keylogging_FULL_Traffic.csv',
               'keylogging_normal': path + 'IoT_Botnet_Dataset_Normal_Keylogging_Traffic.csv',
               'data_exfiltration': path + 'IoT_Botnet_Dataset_Data_Exfiltration_FULL_Traffic.csv',
               'data_exfiltration_normal': path + 'IoT_Botnet_Dataset_Normal_Data_Exfiltration_Traffic.csv',
               }

    # Get the features and category of attack selected for training
    if len(sys.argv) < 4:
        raise ValueError('Please provide the features and attack to train.')

    attack = sys.argv[2].lower()
    features = [f.lower() for f in sys.argv[3:]]

    # Load the dataset
    all_files = [dataset[attack], dataset['normal']]
    cols = ['pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport', 'pkts', 'bytes', 'state', 'ltime',
            'seq', 'dur', 'mean', 'stddev', 'smac', 'dmac', 'sum', 'min', 'max', 'soui', 'doui', 'sco', 'dco', 'spkts',
            'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate', 'attack', 'category', 'subcategory']
    data = pd.concat((pd.read_csv(f, low_memory=False, names=cols) for f in all_files), ignore_index=True)
    # Change columns names to lower case
    data.columns = data.columns.str.lower()

    # Change type of ports columns to avoid errors caused by null values
    data.sport = data.sport.apply(lambda x: -1 if check_port_data(str(x)) else x)
    data.dport = data.dport.apply(lambda x: -1 if check_port_data(str(x)) else x)
    data.sport = data.sport.astype(float)
    data.dport = data.dport.astype(float)

    # Get features and labels
    x = data[features]
    y = data['attack']

    print('Finished data loading')

    return attack, x, y


def create_pre_processor(x, y, attack):
    """
    Function that creates a preprocessing pipeline that can be used to receive the data and map it to the rows
    processed by the trained model. The pipeline is used for training also.
    Categorical features are transformed and oneHotEncoding is applied on them.
    Numerical features are normalized into a range [0,1]
    :param x: features of the preprocessing
    :param y: objective feature
    :param attack: attack name to save the pipeline with trainingAlgorithm-attackName
    :return: data divided for training and the preprocessor
    """
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
    x_train = preprocessor.transform(x_train)  # .toarray()
    print(x_train.shape)

    # Save preprocessor pipeline
    dump(preprocessor, f'../trainResults/preprocessor-{attack}.pkl')

    print('Finished creating preprocessor')
    return x_train, y_train, x_test, y_test, preprocessor

