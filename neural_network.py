import codecs
import numpy as np
import re
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

def preprocess_data(genuine_filepath, bot_filepath):
    # Open csv file and get the tweet part of the csv.
    # Strip out newlines and quotes around text.
    with codecs.open(bot_filepath, 'r', encoding='utf-8', errors='ignore') as bots_file:
        bot_sentences = [x.split(',')[1].strip('\n').strip('"').lower()
                     if len(x.split(',')) > 1 else '' for x in bots_file.readlines()]
    bot_sentences = bot_sentences[1:]
    
    with codecs.open(genuine_filepath, 'r', encoding='utf-8', errors='ignore') as genuine_file:
        genuine_sentences = [x.split(',')[1].strip('\n').strip('"').lower()
                     if len(x.split(',')) > 1 else '' for x in genuine_file.readlines()]
    genuine_sentences = genuine_sentences[1:]

    # Make space between # and hashtag
    bot_sentences.replace('#', '# ')
    genuine_sentences.replace('#', '# ')

    # Split sentences into tokens
    bot_sentences = [sentence.split(' ') for sentence in bot_sentences]
    genuine_sentences = [sentence.split(' ') for sentence in genuine_sentences]

    return bot_sentences, genuine_sentences