from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Dense, Dropout, Input
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
import tensorflow as tf
from joblib import dump, load
import pandas as pd
import time
from load_data import load_data, create_pre_processor

attack, x, y = load_data()
x_train, y_train, x_test, y_test, preprocessor = create_pre_processor(x, y, attack)


# Define Neural Network
def train_nn(nn1=512, nn2=128, dropout=0.5, lr=0.001, hidden_activation='relu'):
    nn = Sequential(name='nn_attacks')
    output = 1
    # Input layer
    nn.add(Input(batch_input_shape=(None,  x_train.shape[1])))
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
dump(grid_search.best_estimator_['clf'].history_, f'../trainResults/history-rn-{attack}.pkl')


# Load best model
loaded_model = tf.keras.models.load_model(f'../bestModels/rn-{attack}.h5')

# Load the preprocessor and transform the test data
loaded_preprocessor = load(f'../trainResults/preprocessor-{attack}.pkl')
p_x_test = preprocessor.transform(x_test) # .toarray()

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

print('Script Finished!')
