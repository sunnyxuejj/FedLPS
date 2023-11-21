"""Builds vocabulary file from data."""

import argparse
import collections
import json
import os
import pickle

def user_data_collect(train_data, client_num, vocab):
    train_tokens = {}
    val_data = {}
    test_data = {}
    count = {}
    for u in train_data:
        sents = []
        for i in range(len(train_data[u]['x'])):
            sents.extend(train_data[u]['x'][i])
        for i in range(len(sents)):
            for j in range(len(sents[i])):
                if not sents[i][j] in vocab:
                    sents[i][j] = '<UNK>'
        count[u] = {'num': len(sents), 'sents': sents}
    count = sorted(count.items(), key=lambda x: x[1]['num'], reverse=True)
    for i in range(client_num):
        sents_num = len(count[i][1]['sents'])
        train_size = int(sents_num * 0.8)
        val_size = int(sents_num * 0.9)
        train_tokens[i] = count[i][1]['sents'][:train_size]
        val_data[i] = count[i][1]['sents'][train_size: val_size]
        test_data[i] = count[i][1]['sents'][val_size: -1]
    return train_tokens, val_data, test_data


def load_leaf_data(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        to_ret = data['user_data']
        data = None
    return to_ret

def build_counter(train_data, initial_counter=None):
    train_tokens = []
    for u in train_data:
        for c in train_data[u]['x']:
            train_tokens.extend([s for s in c])

    all_tokens = []
    for i in train_tokens:
        all_tokens.extend(i)
    train_tokens = []

    if initial_counter is None:
        counter = collections.Counter()
    else:
        counter = initial_counter

    counter.update(all_tokens)
    all_tokens = []

    return counter #coubtor包含每个token出现的次数共33016个词


def build_vocab(counter, vocab_size=10000):
    pad_symbol, unk_symbol = 0, 1
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    count_pairs = count_pairs[:(vocab_size - 2)] # -2 to account for the unknown and pad symbols

    words, _ = list(zip(*count_pairs))

    vocab = {}
    vocab['<PAD>'] = pad_symbol
    vocab['<UNK>'] = unk_symbol

    for i, w in enumerate(words):
        if w != '<PAD>':
            vocab[w] = i + 1

    return {'vocab': vocab, 'size': vocab_size, 'unk_symbol': unk_symbol, 'pad_symbol': pad_symbol}

def save_vocab(vocab, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    pickle.dump(vocab, open(os.path.join(target_dir, 'reddit_vocab.pck'), 'wb'))


def data_process(data_dir, vocab_size, client_num):
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    json_files.sort()

    for f in json_files:
        print('loading {}'.format(f))
        data = load_leaf_data(os.path.join(data_dir, f))
        if not os.path.exists('reddit_vocab.pck'):
            counter = build_counter(data)
            vocab = build_vocab(counter, vocab_size=vocab_size)  # 词汇表是一个字典
            save_vocab(vocab, '../')
        else:
            with open('reddit_vocab.pck', 'rb') as f:
                vocab = pickle.load(f)
        train_data, val_data, test_data = user_data_collect(data, client_num, vocab['vocab'])
    return train_data, val_data, test_data

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir',
        default= './train',
        help='dir with training file;',
        type=str,
        required=False)
    parser.add_argument('--vocab-size',
        help='size of the vocabulary;',
        type=int,
        default=10000,
        required=False)
    parser.add_argument('--target-dir',
        help='dir with training file;',
        type=str,
        default='./',
        required=False)
    parser.add_argument('--client_num', type=int, default=100, required=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_data, val_data, test_data = data_process(args.data_dir, args.vocab_size, 100)
