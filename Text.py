#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8
from torch.utils.data import Dataset

class DatasetLM(Dataset):
    def __init__(self, dataset, vocab):
        """ Loads the data at the given path using the given index (maps tokens to indices).
        Returns a list of sentences where each is a list of token indices.
        """
        self.list_sent = []
        for sentence in dataset:
            sent = []
            for token in sentence:
                sent.append(vocab[token])
            self.list_sent.append(sent)

    def __len__(self):
        return len(self.list_sent)

    def __getitem__(self, idx):
        return self.list_sent[idx]
