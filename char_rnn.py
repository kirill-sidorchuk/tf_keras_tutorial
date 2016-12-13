import argparse
import os
from random import random

import numpy as np

from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import Sequential


def load_list_from_file(file_name):
    lines = []
    f = open(file_name, 'r')
    for line in f:
        lines.append(line.strip())
    f.close()
    return lines


def extract_set_of_chars(list_of_files):
    all_chars = set()
    for file in list_of_files:
        text = open(file, 'r').read()
        chars = set(text)
        all_chars |= chars
    return all_chars


def create_model(chars, max_len):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(max_len, len(chars))))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def get_dir_files(dir):
    dir_list = list()
    files = os.listdir(dir)
    for file in files:
        dir_list.append(os.path.join(dir, file))
    return dir_list


def prepare_data(inputs, max_len, chars, char_labels, outputs):
    # using bool to reduce memory usage
    X = np.zeros((len(inputs), max_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(inputs), len(chars)), dtype=np.bool)

    # set the appropriate indices to 1 in each one-hot vector
    for i, example in enumerate(inputs):
        for t, char in enumerate(example):
            X[i, t, char_labels[char]] = 1
        y[i, char_labels[outputs[i]]] = 1

    return X, y


def load_all_text(files):
    text = list()
    for file in files:
        this_text = open(file, 'r').read()
        text += this_text
    return text


def generate(max_len, text, chars, char_labels, labels_char, model,
             temperature=0.35, seed=None, predicate=lambda x: len(x) < 100):
    if seed is None and len(seed) < max_len:
        raise Exception('Seed text must be at least {} chars long'.format(max_len))

    # if no seed text is specified, randomly select a chunk of text
    else:
        start_idx = random.randint(0, len(text) - max_len - 1)
        seed = text[start_idx:start_idx + max_len]

    sentence = seed
    generated = sentence

    while predicate(generated):
        # generate the input tensor
        # from the last max_len characters generated so far
        x = np.zeros((1, max_len, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_labels[char]] = 1.

        # this produces a probability distribution over characters
        probs = model.predict(x, verbose=0)[0]

        # sample the character to use based on the predicted probabilities
        next_idx = sample(probs, temperature)
        next_char = labels_char[next_idx]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated


def sample(probs, temperature):
    """samples an index from a vector of probabilities"""
    a = np.log(probs)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def train(args):

    train_files = get_dir_files(args.train_dir)
    print('Train corpus contains %d files' % len(train_files))

    train_chars_set = extract_set_of_chars(train_files)
    print('Train set of chars contains %d different chars' % len(train_chars_set))

    test_files = get_dir_files(args.test_dir)
    print('Test corpus contains %d files' % len(test_files))

    test_chars_set = extract_set_of_chars(test_files)
    print('Test set of chars contains %d different chars' % len(test_chars_set))

    all_chars = list(test_chars_set | train_chars_set)
    print('Total number of different chars = %d' % len(train_chars_set))

    char_labels = {ch: i for i, ch in enumerate(all_chars)}
    labels_char = {i: ch for i, ch in enumerate(all_chars)}

    text = load_all_text(train_files)
    print('Training text has %1.2fM chars' % float(len(text)*1e-6))

    # text window length
    max_len = 20

    step = 3
    inputs = []
    outputs = []
    for i in range(0, len(text) - max_len, step):
        inputs.append(text[i:i + max_len])
        outputs.append(text[i + max_len])

    X, y = prepare_data(inputs, max_len, all_chars, char_labels, outputs)

    model = create_model(all_chars, max_len)

    n_epochs = 10
    for i in range(n_epochs):
        print('epoch', i)

        # set nb_epoch to 1 since we're iterating manually
        model.fit(X, y, batch_size=128, nb_epoch=1)

        # preview
        for temp in [0.2, 0.5, 1., 1.2]:
            print('\n\ttemperature:', temp)
            print(generate(max_len, text, all_chars, char_labels, labels_char, model, temperature=temp))

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train char RNN.')
    parser.add_argument("train_dir", type=str, help="directory with train text files")
    parser.add_argument("test_dir", type=str, help="directory with test text files")
    _args = parser.parse_args()
    train(_args)
