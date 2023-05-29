from joblib import load
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from load_data import load_data, create_pre_processor

# load the data
attack, x, y = load_data()
x_train, y_train, x_test, y_test, pp = create_pre_processor(x, y, attack, dump_preprocessor=False)
# Load the preprocessor
loaded_preprocessor = load(f'../trainResults/preprocessor-{attack}.pkl')


def attack_latency(test_data, model, algorithm):
    w = np.arange(1000, len(test_data), int(len(test_data)/100))
    times = [0] * len(w)
    times_iterator = 0
    for i in w: 
        to_predict = test_data[:i]
        print(f'Predicting {len(to_predict)} samples with {algorithm}...')
        init = time.time()
        to_predict = loaded_preprocessor.transform(to_predict)
        model.predict(to_predict)
        end = time.time()
        times[times_iterator] = end - init
        times_iterator += 1

    print(f'Finished predicting {len(test_data)} samples with {algorithm}!')
    plt.plot(w, times)
    plt.xlabel('Number of samples')
    plt.ylabel('Time elapsed (s)')
    plt.title(f'Latency of {algorithm} with {attack}')
    plt.savefig(f"../testResults/latency/{algorithm}-{attack}-image.png")
    plt.clf()
    print(f'Latency test for {algorithm} finished!')


try:
    # Load SVM best model
    loaded_model = load(f'../bestModels/svm-{attack}.joblib')
    attack_latency(x_test, loaded_model, 'SVM')
except Exception as e:
    print(e)
    print(f'SVM model for {attack} not found')

try:
    # Load best model autoencoder
    loaded_model = tf.keras.models.load_model(f'../bestModels/autoencoder-{attack}.h5')
    attack_latency(x_test, loaded_model, 'Autoencoder')
except Exception as e:
    print(f'Autoencoder model for {attack} not found')
    print(e)

try: 
    # Load best model for RN 
    loaded_model = tf.keras.models.load_model(f'../bestModels/rn-{attack}.h5')
    attack_latency(x_test, loaded_model, 'RN')
except Exception as e:
    print(e)
    print(f'RN model for {attack} not found')

print(f'Latency test for {attack} finished!')
