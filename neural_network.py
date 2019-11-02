import numpy as np
import time

from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense

def train_model(training_data, training_labels, evaluation_data, evaluation_labels):
    model = Sequential()
    model.add(Dense(200, input_dim=1000, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(np.array(training_data), np.array(training_labels), epochs=150, batch_size=10)

    _, accuracy = model.evaluate(np.array(evaluation_data), np.array(evaluation_labels))
    print('Accuracy: %.2f' % (accuracy*100))

    model.save('model-{}.h5'.format(int(time.time())))
