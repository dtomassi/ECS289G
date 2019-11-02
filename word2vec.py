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

def train_word2vec_model(fp_list_tuple):
    # for fp_tuple in fp_list_tuple:
    pass


def main(argv=None):
    argv = argv or sys.argv
    reports_dir = _validate_input(argv)


def _print_usage():
    print('Usage: python3 spotbugs_parser.py <reports_dir>')
    print('reports_dir: Path to the directory of reports')


def _validate_input(argv):
    if len(argv) != 2:
        _print_usage()
        sys.exit(1)
    reports_dir = argv[1]
    if not os.path.isdir(reports_dir) and os.path.exists(reports_dir):
        print('The reports_dir argument is not a file or does not exist. Exiting.')
        _print_usage()
        sys.exit(1)
    return reports_dir


if __name__ == '__main__':
    sys.exit(main())
