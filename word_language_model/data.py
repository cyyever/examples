import os

import torch
import torchtext


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self):
        self.dictionary = Dictionary()
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

        train_dataset = torchtext.datasets.WikiText2(root=data_dir, split="train")
        valid_dataset = torchtext.datasets.WikiText2(root=data_dir, split="valid")
        test_dataset = torchtext.datasets.WikiText2(root=data_dir, split="test")
        self.train = self.tokenize(train_dataset)
        self.valid = self.tokenize(valid_dataset)
        self.test = self.tokenize(test_dataset)

    def tokenize(self, dataset):
        """Tokenizes a dataset."""
        # Add words to the dictionary
        for line in dataset:
            words = line.split() + ["<eos>"]
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize file content
        idss = []
        for line in dataset:
            words = line.split() + ["<eos>"]
            ids = []
            for word in words:
                ids.append(self.dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)
        return ids
