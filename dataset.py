"""Dataset utilities for Multi30k German→English translation.
Handles loading, tokenization (spacy), vocabulary construction,
and numericalization of sentence pairs.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy


# Fixed across all experiments
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
SPECIALS = ['<unk>', '<pad>', '<sos>', '<eos>']


class Vocabulary:
    """Lightweight word↔index mapping with special-token support."""

    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}

    def build(self, token_lists, min_freq=1):
        """Build vocab from an iterable of token lists.

        Args:
            token_lists : list[list[str]] — tokenized sentences.
            min_freq    : int — minimum frequency to include a token.
        """
        freq = {}
        for tokens in token_lists:
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
        for idx, tok in enumerate(SPECIALS):
            self.token_to_idx[tok] = idx
            self.idx_to_token[idx] = tok

        idx = len(SPECIALS)
        for tok in sorted(freq.keys()):
            if tok not in self.token_to_idx and freq[tok] >= min_freq:
                self.token_to_idx[tok] = idx
                self.idx_to_token[idx] = tok
                idx += 1

    def __len__(self):
        return len(self.token_to_idx)

    def encode(self, tokens):
        """Convert token list → index list (with <sos>/<eos>)."""
        return (
            [SOS_IDX]
            + [self.token_to_idx.get(t, UNK_IDX) for t in tokens]
            + [EOS_IDX]
        )

    def decode(self, indices, strip_special=True):
        """Convert index list → token list."""
        tokens = [self.idx_to_token.get(i, '<unk>') for i in indices]
        if strip_special:
            tokens = [t for t in tokens if t not in SPECIALS]
        return tokens


class Multi30kDataset(Dataset):
    """PyTorch Dataset wrapping one split of Multi30k (de→en)."""

    def __init__(self, data_pairs, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
        """
        Args:
            data_pairs    : list of (de_sentence, en_sentence) strings.
            src_vocab     : Vocabulary for German.
            tgt_vocab     : Vocabulary for English.
            src_tokenizer : spacy tokenizer callable for German.
            tgt_tokenizer : spacy tokenizer callable for English.
        """
        self.data          = data_pairs
        self.src_vocab     = src_vocab
        self.tgt_vocab     = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        src_tokens = [t.text.lower() for t in self.src_tokenizer(src_text)]
        tgt_tokens = [t.text.lower() for t in self.tgt_tokenizer(tgt_text)]
        src_indices = torch.tensor(self.src_vocab.encode(src_tokens), dtype=torch.long)
        tgt_indices = torch.tensor(self.tgt_vocab.encode(tgt_tokens), dtype=torch.long)
        return src_indices, tgt_indices


def collate_fn(batch):
    """Pad variable-length src/tgt sequences in a batch."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def load_tokenizers():
    """Load spacy tokenizers for German and English."""
    de_tok = spacy.load('de_core_news_sm')
    en_tok = spacy.load('en_core_web_sm')
    return de_tok, en_tok


def prepare_data(min_freq=2):
    """Load Multi30k, build vocabs, and return datasets + vocabs.

    Returns:
        train_ds, val_ds, test_ds : Multi30kDataset instances.
        src_vocab, tgt_vocab      : Vocabulary instances.
        de_tok, en_tok             : spacy tokenizer objects.
    """
    # load from HuggingFace
    raw = load_dataset('bentrevett/multi30k')

    de_tok, en_tok = load_tokenizers()

    # extract pairs per split
    def extract_pairs(split_data):
        return [(ex['de'], ex['en']) for ex in split_data]

    train_pairs = extract_pairs(raw['train'])
    val_pairs   = extract_pairs(raw['validation'])
    test_pairs  = extract_pairs(raw['test'])

    # tokenize training set
    src_token_lists = [[t.text.lower() for t in de_tok(pair[0])] for pair in train_pairs]
    tgt_token_lists = [[t.text.lower() for t in en_tok(pair[1])] for pair in train_pairs]

    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build(src_token_lists, min_freq=min_freq)
    tgt_vocab.build(tgt_token_lists, min_freq=min_freq)
    train_ds = Multi30kDataset(train_pairs, src_vocab, tgt_vocab, de_tok, en_tok)
    val_ds   = Multi30kDataset(val_pairs,   src_vocab, tgt_vocab, de_tok, en_tok)
    test_ds  = Multi30kDataset(test_pairs,  src_vocab, tgt_vocab, de_tok, en_tok)

    return train_ds, val_ds, test_ds, src_vocab, tgt_vocab, de_tok, en_tok