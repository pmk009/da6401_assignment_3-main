"""Dataset utilities for Multi30k German→English translation.
Handles loading, tokenization (spacy), vocabulary construction,
and numericalization of sentence pairs.

Caches processed data to disk so subsequent runs skip HuggingFace
download and spacy tokenization entirely.
"""

import os
import pickle
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

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data_cache')


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
    """PyTorch Dataset with pre-numericalized (de→en) sentence pairs."""

    def __init__(self, src_indices_list, tgt_indices_list):
        """
        Args:
            src_indices_list : list of 1-D LongTensors (pre-encoded src sentences).
            tgt_indices_list : list of 1-D LongTensors (pre-encoded tgt sentences).
        """
        self.src = src_indices_list
        self.tgt = tgt_indices_list

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


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


def _tokenize_pairs(pairs, de_tok, en_tok):
    """Tokenize a list of (de, en) string pairs with spacy.

    Returns:
        src_token_lists, tgt_token_lists : list[list[str]]
    """
    src_tokens = [[t.text.lower() for t in de_tok(p[0])] for p in pairs]
    tgt_tokens = [[t.text.lower() for t in en_tok(p[1])] for p in pairs]
    return src_tokens, tgt_tokens


def _encode_split(token_lists, vocab):
    """Numericalize a list of token lists into a list of LongTensors."""
    return [torch.tensor(vocab.encode(tokens), dtype=torch.long) for tokens in token_lists]


def prepare_data(min_freq=2):
    """Load Multi30k, build vocabs, and return datasets + vocabs.

    On first call, downloads from HuggingFace, tokenizes with spacy,
    builds vocabs, numericalizes everything, and caches to .data_cache/.
    On subsequent calls, loads directly from cache (~instant).

    Returns:
        train_ds, val_ds, test_ds : Multi30kDataset instances.
        src_vocab, tgt_vocab      : Vocabulary instances.
        de_tok, en_tok            : spacy tokenizer objects.
    """
    cache_file = os.path.join(CACHE_DIR, f'multi30k_mf{min_freq}.pkl')

    if os.path.exists(cache_file):
        print(f"  Loading cached data from {cache_file} ...")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)

        src_vocab = cached['src_vocab']
        tgt_vocab = cached['tgt_vocab']

        train_ds = Multi30kDataset(cached['train_src'], cached['train_tgt'])
        val_ds   = Multi30kDataset(cached['val_src'],   cached['val_tgt'])
        test_ds  = Multi30kDataset(cached['test_src'],  cached['test_tgt'])

        de_tok, en_tok = load_tokenizers()
        return train_ds, val_ds, test_ds, src_vocab, tgt_vocab, de_tok, en_tok

    
    raw = load_dataset('bentrevett/multi30k')
    de_tok, en_tok = load_tokenizers()

    def extract_pairs(split_data):
        return [(ex['de'], ex['en']) for ex in split_data]

    train_pairs = extract_pairs(raw['train'])
    val_pairs   = extract_pairs(raw['validation'])
    test_pairs  = extract_pairs(raw['test'])

    # tokenize all splits
    train_src_tok, train_tgt_tok = _tokenize_pairs(train_pairs, de_tok, en_tok)
    val_src_tok,   val_tgt_tok   = _tokenize_pairs(val_pairs,   de_tok, en_tok)
    test_src_tok,  test_tgt_tok  = _tokenize_pairs(test_pairs,  de_tok, en_tok)

    # build vocabs from training data only
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build(train_src_tok, min_freq=min_freq)
    tgt_vocab.build(train_tgt_tok, min_freq=min_freq)

    # numericalize
    train_src = _encode_split(train_src_tok, src_vocab)
    train_tgt = _encode_split(train_tgt_tok, tgt_vocab)
    val_src   = _encode_split(val_src_tok,   src_vocab)
    val_tgt   = _encode_split(val_tgt_tok,   tgt_vocab)
    test_src  = _encode_split(test_src_tok,  src_vocab)
    test_tgt  = _encode_split(test_tgt_tok,  tgt_vocab)

    # cache to disk
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab,
            'train_src': train_src, 'train_tgt': train_tgt,
            'val_src':   val_src,   'val_tgt':   val_tgt,
            'test_src':  test_src,  'test_tgt':  test_tgt,
        }, f)
    print(f"  Cached processed data → {cache_file}")

    train_ds = Multi30kDataset(train_src, train_tgt)
    val_ds   = Multi30kDataset(val_src,   val_tgt)
    test_ds  = Multi30kDataset(test_src,  test_tgt)

    return train_ds, val_ds, test_ds, src_vocab, tgt_vocab, de_tok, en_tok