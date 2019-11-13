import codecs
import numpy as np
import random
import re
import time

from datetime import datetime
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers

def train_model(training_data, training_labels, evaluation_data, evaluation_labels):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(2000,)))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(np.array(training_data), np.array(training_labels), epochs=100, batch_size=10)

    _, accuracy = model.evaluate(np.array(evaluation_data), np.array(evaluation_labels))
    print('Accuracy: %.2f' % (accuracy*100))
    print('time:',int(time.time()))
    model.save('model-{}.h5'.format(int(time.time())))

def preprocess_data(genuine_filepath, bot_filepath):
    """
    Preprocess data and normalize tweets.
    """
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

    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
            'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
            'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens
        
        # corpus from which the word statistics are going to be used 
        # for word segmentation 
        segmenter="twitter", 
        
        # corpus from which the word statistics are going to be used 
        # for spell correction
        corrector="twitter", 
        
        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words
        
        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        
        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )
    bot_sentences = [text_processor.pre_process_doc(s) for s in bot_sentences]
    genuine_sentences = [text_processor.pre_process_doc(s) for s in genuine_sentences]

    return genuine_sentences, bot_sentences


def segment_data(pos_examples, neg_examples):
    pos_size, neg_size = len(pos_examples) - 1, len(neg_examples) - 1 
    pos_indx, neg_indx = int(pos_size * .9), int(neg_size * .9)

    pos_tuples = [(s, 1) for s in pos_examples]
    neg_tuples = [(s, 0) for s in neg_examples]
    
    training_tuples = [*pos_tuples[:pos_indx], *neg_tuples[:neg_indx]]
    eval_tuples = [*pos_tuples[pos_indx:], *neg_tuples[neg_indx:]]
    random.shuffle(training_tuples)
    random.shuffle(eval_tuples)

    return training_tuples, eval_tuples
