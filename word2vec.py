import os
import sys


def create_training_data(tokenized_sentences, word2vec_model, pos_or_neg, desired_length):
    '''
    Creates training data by returing the embedded version of the sentence with the label.
    Currently only using tweets of desired_length words or less.
    
    :tokenized_sentences: list of list of words
    :word2vec_model: trained word2vec model
    :pos_or_neg: 0 if genuine, 1 if bot
    
    Returns:
    X: list of embedded sentences
    Y: List of Labels
    '''
    pruned_embedded_sentences = []
    for sentence in tokenized_sentences:
        embedded_sentence = list()
        for word in sentence:
            if word not in word2vec_model:
                embedded_sentence = None
                break
            embedded_sentence = [*embedded_sentence, *word2vec_model[word]]
        if embedded_sentence is not None:
            if len(embedded_sentence) != desired_length*100:
                diff = desired_length*100 - len(embedded_sentence)
                embedded_sentence += [0]*diff
            pruned_embedded_sentences.append(embedded_sentence) 

    return pruned_embedded_sentences, [pos_or_neg] * len(pruned_embedded_sentences)
