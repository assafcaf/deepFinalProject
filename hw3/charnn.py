import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator
import numpy as np
import copy
import torch.nn.functional as F
from torch.nn import GRU
def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    chrs = np.unique(list(text))
    char_to_idx = {chrs[i].item(): i for i in range(len(chrs))}
    idx_to_char = {i: chrs[i].item() for i in range(len(chrs))}
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    text_clean = copy.deepcopy(text)
    for chr in chars_to_remove:
        text_clean = text_clean.replace(chr, '')
    n_removed = len(text) - len(text_clean)
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    result = torch.zeros(len(text), len(char_to_idx), dtype=torch.int8)
    for i, chr in enumerate(text):
        result[i, char_to_idx[chr]] = 1
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    result = ""
    for i in range(embedded_text.shape[0]):
        result += idx_to_char[embedded_text[i].argmax().item()]
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    embedded_text = chars_to_onehot(text, char_to_idx)
    upto = seq_len * ((len(embedded_text) - 1) // seq_len)

    # remove last char
    samples = embedded_text[:-1]
    labels = embedded_text[1:].argmax(dim=1)

    # split to groups of seq_len
    samples = samples[:upto].reshape(-1, seq_len, len(char_to_idx)).to(device)
    labels = labels[:upto].reshape(-1, seq_len).to(device)

    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    return F.softmax(y / temperature, dim=dim)


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence
    model.eval()
    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    with torch.no_grad():
        hidden_state = None
        pred = start_sequence
        for _ in range(n_chars - len(start_sequence)):
            x = chars_to_onehot(pred, char_to_idx).to(device).unsqueeze(0).to(torch.float32)
            logits, hidden_state = model(x, hidden_state)
            hidden_state = hidden_state.detach()
            probs = hot_softmax(logits[0, -1, :], temperature=T)
            pred = idx_to_char[torch.multinomial(probs, 1).item()]
            out_text += pred
    model.train()
    return out_text

class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.window_size = len(dataset) // batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.

        return iter([self.window_size * i + j for j in range(self.window_size)
                     for i in range(len(self.dataset) // self.window_size)])


    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.

        # TODO: Create the parameters of the model for all layers.

        self.layers = self.build_layers()
        self.output_layer = nn.Linear(h_dim, out_dim)

    def build_layers(self):
        if self.n_layers == 1:
            return nn.Sequential(UnitGRU(self.in_dim, self.h_dim))

        layers = [UnitGRU(self.in_dim, self.h_dim, dropout=self.dropout)]
        for i in range(self.n_layers-2):
            layers.append(UnitGRU(self.h_dim, self.h_dim, dropout=self.dropout))
        layers.append(UnitGRU(self.h_dim, self.h_dim))
        return nn.Sequential(*layers)

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        if hidden_state is None:
            hidden_state = torch.zeros((self.n_layers, input.shape[0], self.h_dim), device=input.device)
        else:
            hidden_state = hidden_state.transpose(0, 1)
        x = input
        hidden_states = []
        for layer, h in zip(self.layers, hidden_state):
            out_put, x = layer(x, h)
            hidden_states.append(out_put)

        out_put = self.output_layer(x)
        return out_put, torch.stack(hidden_states).transpose(0, 1)


class UnitGRU(nn.Module):
    def __init__(self, in_dim, h_dim, batch_first=True, dropout=0):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.layer_params = []
        self.batch_first = batch_first

        #  reset gate parameters (r)
        self.W_xr = nn.Linear(in_dim, h_dim, bias=True)
        self.W_hr = nn.Linear(h_dim, h_dim,  bias=False)
        self.add_module('W_xr', self.W_xr)
        self.add_module('W_hr', self.W_hr)

        #  update gate parameters (z)
        self.W_xz = nn.Linear(in_dim, h_dim,  bias=True)
        self.W_hz = nn.Linear(h_dim, h_dim,  bias=False)
        self.add_module('W_hz', self.W_hz)
        self.add_module('W_xz', self.W_xz)

        # input new gate parameters (n)
        self.W_xn = nn.Linear(in_dim, h_dim,  bias=True)
        self.W_hn = nn.Linear(h_dim, h_dim,  bias=False)
        self.add_module('W_hn', self.W_hn)
        self.add_module('W_xn', self.W_xn)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, hidden_state: Tensor = None):
        if hidden_state is None:
            hidden_state = torch.zeros(self.h_dim, device=inputs.device)
        hidden_states = []

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        for xt in inputs:
            r = torch.sigmoid(self.W_xr(xt) + self.W_hr(hidden_state))
            z = torch.sigmoid(self.W_xz(xt) + self.W_hz(hidden_state))
            n = torch.tanh(self.W_xn(xt) + (r * self.W_hn(hidden_state)))
            hidden_state = ((1 - z) * n) + (z * hidden_state)
            hidden_states.append(hidden_state)

        hidden_states = torch.stack(hidden_states)
        if self.batch_first:
            hidden_states = hidden_states.transpose(0, 1)

        hidden_state = self.dropout(hidden_state)
        return hidden_state, hidden_states
