import codecs

from neural_network import train_model
from word2vec import create_training_data
from gensim.models import Word2Vec

bots_fp = '/home/dtomassi/tweets_dataset/datasets_full.csv/social_spambots_2.csv/tweets.csv'
genuine_fp = '/home/dtomassi/tweets_dataset/datasets_full.csv/genuine_accounts.csv/tweets.csv'
eval_fp = '/home/dtomassi/tweets_dataset/datasets_full.csv/traditional_spambots_1.csv/tweets.csv'

with codecs.open(bots_fp, 'r', encoding='utf-8', errors='ignore') as bots_file:
    bot_sentences = [x.split(',')[1].strip('\n').strip('"').lower()
                 if len(x.split(',')) > 1 else '' for x in bots_file.readlines()]
bot_sentences = bot_sentences[1:]
with codecs.open(genuine_fp, 'r', encoding='utf-8', errors='ignore') as genuine_file:
    genuine_sentences = [x.split(',')[1].strip('\n').strip('"').lower()
                 if len(x.split(',')) > 1 else '' for x in genuine_file.readlines()]
genuine_sentences = genuine_sentences[1:]
with codecs.open(eval_fp, 'r', encoding='utf-8', errors='ignore') as eval_file:
    eval_sentences = [x.split(',')[1].strip('\n').strip('"').lower()
                 if len(x.split(',')) > 1 else '' for x in eval_file.readlines()]
eval_sentences = genuine_sentences[1:]
print('Read in files')

pos_tokenizes_sentences = [sentence.split(' ') for sentence in bot_sentences][:400000]
neg_tokenizes_sentences = [sentence.split(' ') for sentence in genuine_sentences][:600000]
eval_tokenizes_sentences = [sentence.split(' ') for sentence in eval_sentences][:600000]
print('Tokenized sentences')

neg_tokenizes_sentences_10 = list(filter(lambda x: len(x) <= 10, neg_tokenizes_sentences))
pos_tokenizes_sentences_10 = list(filter(lambda x: len(x) <= 10, pos_tokenizes_sentences))
eval_tokenizes_sentences_10 = list(filter(lambda x: len(x) <= 10, eval_tokenizes_sentences))
print('Filtered tweets longer than 10 words')

word2vec_model = Word2Vec.load("word2vec.model")
neg_training_data, neg_training_labels = create_training_data(neg_tokenizes_sentences_10, word2vec_model, 0, 10)
pos_training_data, pos_training_labels = create_training_data(pos_tokenizes_sentences_10, word2vec_model, 1, 10)
eval_training_data, eval_training_labels = create_training_data(eval_tokenizes_sentences_10, word2vec_model, 1, 10)
print('Generated training and evaluation data')

training_data = [*pos_training_data, *neg_training_data]
training_labels = [*pos_training_labels, *neg_training_labels]

print('Training Model')
train_model(training_data, training_labels, eval_training_data, eval_training_labels)
print('Finished training model')