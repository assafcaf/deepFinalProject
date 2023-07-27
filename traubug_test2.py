import unittest
import os
import sys
import pathlib
import urllib
import shutil
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
import hw3.charnn as charnn
from hw3.charnn import SequenceBatchSampler

corpus_path = r'C:\Users\Assaf\.pytorch-datasets\test.txt'
test = unittest.TestCase()
plt.rcParams.update({'font.size': 12})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print('Using device:', device)


# params
seq_len = 64
h_dim = 64
n_layers = 2
batch_size = 16
dropout = 0.4

with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus = f.read()

char_to_idx, idx_to_char = charnn.char_maps(corpus)
corpus, n_removed = charnn.remove_chars(corpus, ['}','$','_','<','\ufeff'])
# Wrap the actual embedding functions for calling convenience
def embed(text):
    return charnn.chars_to_onehot(text, char_to_idx)

def unembed(embedding):
    return charnn.onehot_to_chars(embedding, idx_to_char)


vocab_len = len(char_to_idx)
num_samples = (len(corpus) - 1) // seq_len
samples, labels = charnn.chars_to_labelled_samples(corpus, char_to_idx, seq_len, device)


ds_corpus = torch.utils.data.TensorDataset(samples, labels)
sampler_corpus = SequenceBatchSampler(ds_corpus, batch_size)
dl_corpus = torch.utils.data.DataLoader(ds_corpus, batch_size=batch_size, sampler=sampler_corpus, shuffle=False)

# Pick a tiny subset of the dataset
subset_start, subset_end = 0, 4
ds_corpus_ss = torch.utils.data.Subset(ds_corpus, range(subset_start, subset_end))
batch_size_ss = 1
sampler_ss = SequenceBatchSampler(ds_corpus_ss, batch_size=batch_size_ss)
dl_corpus_ss = torch.utils.data.DataLoader(ds_corpus_ss, batch_size_ss, sampler=sampler_ss, shuffle=False)

# Convert subset to text
subset_text = ''
for i in range(subset_end - subset_start):
    subset_text += unembed(ds_corpus_ss[i][0])
print(f'Text to "memorize":\n\n{subset_text}')

import torch.nn as nn
import torch.optim as optim
from hw3.training import RNNTrainer

torch.manual_seed(42)

from hw3.answers import part1_rnn_hyperparams
in_dim = vocab_len
lr = 0.01
num_epochs = 100
model = charnn.MultilayerGRU(in_dim, h_dim, out_dim=in_dim, n_layers=n_layers).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
trainer = RNNTrainer(model, loss_fn, optimizer, device)

for epoch in range(num_epochs):
    epoch_result = trainer.train_epoch(dl_corpus_ss, verbose=False)

    # Every X epochs, we'll generate a sequence starting from the first char in the first sequence
    # to visualize how/if/what the model is learning.
    if epoch == 0 or (epoch + 1) % 25 == 0:
        avg_loss = np.mean(epoch_result.losses)
        accuracy = np.mean(epoch_result.accuracy)
        print(f'\nEpoch #{epoch + 1}: Avg. loss = {avg_loss:.3f}, Accuracy = {accuracy:.2f}%')

        generated_sequence = charnn.generate_from_model(model, subset_text[0],
                                                        seq_len * (subset_end - subset_start),
                                                        (char_to_idx, idx_to_char), T=0.1)

        # Stop if we've successfully memorized the small dataset.
        print(generated_sequence)
        if generated_sequence == subset_text:
            break

# Test successful overfitting
test.assertGreater(epoch_result.accuracy, 99)
test.assertEqual(generated_sequence, subset_text)



