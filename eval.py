import ast
import codecs

from neural_network import train_model
from neural_network import segment_data
from word2vec import create_training_data
from gensim.models import Word2Vec

LENGTH_TWEETS = 20
bots_fp = '/home/dtomassi/github/ECS289G/bot_tokenized_sentences.txt'
gen_fp = '/home/dtomassi/github/ECS289G/gen_tokenized_sentences.txt'

with open(bots_fp) as bots_file:
    pos_tokenizes_sentences = [ast.literal_eval(x.strip()) for x in bots_file.readlines()]

with open(gen_fp) as gen_file:
    neg_tokenizes_sentences = [ast.literal_eval(x.strip()) for x in gen_file.readlines()]

pruned_pos_tokenizes_sentences = []
for sentence in pos_tokenizes_sentences:
    if sentence != []:
        pruned_pos_tokenizes_sentences.append(sentence)

pruned_neg_tokenizes_sentences = []
for sentence in neg_tokenizes_sentences:
    if sentence != []:
        pruned_neg_tokenizes_sentences.append(sentence)

neg_tokenizes_sentences_10 = list(filter(lambda x: len(x) <= LENGTH_TWEETS, pruned_neg_tokenizes_sentences))
pos_tokenizes_sentences_10 = list(filter(lambda x: len(x) <= LENGTH_TWEETS, pruned_pos_tokenizes_sentences))
print('Filtered tweets longer than 10 words')

word2vec_model = Word2Vec.load("word2vec-1573008850.model")
neg_data, neg_labels = create_training_data(neg_tokenizes_sentences_10, word2vec_model, 0, LENGTH_TWEETS)
pos_data, pos_labels = create_training_data(pos_tokenizes_sentences_10, word2vec_model, 1, LENGTH_TWEETS)
print('Generated training and evaluation data')

training_tuple, eval_tuple = segment_data(pos_data, neg_data)
training_tuple_sep, eval_tuple_sep = list(map(list, zip(*training_tuple))), list(map(list, zip(*eval_tuple)))
training_data, training_labels = training_tuple_sep[0], training_tuple_sep[1]
eval_training_data, eval_training_labels = eval_tuple_sep[0], eval_tuple_sep[1]
print('Training Model')
train_model(training_data[:int(len(training_data)/4)], training_labels[:int(len(training_labels)/4)], eval_training_data, eval_training_labels)
print('Finished training model')
