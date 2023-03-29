# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import sys
import os
import numpy as np
from Bio import pairwise2

from torch.utils.data import Dataset
import dask
import dask.dataframe as dd
import dask.array as da
import dask.bag as db


class miRNA_CTS_Dask_Dataset(Dataset):
    """ PyTorch dataloader for miRNA-CTS pair data using Dask """
    def __init__(self, X, labels, set_idxs, set_labels):
        self.X = X
        self.labels = labels.astype(np.int64)
        self.set_idxs = set_idxs
        self.set_labels = set_labels

    def __getitem__(self, i):
        x, label, set_idx = dask.compute(self.X[i], self.labels[i], self.set_idxs[i], scheduler='threads')
        return x, label, set_idx


    def __len__(self):
        return self.labels.shape[0]

    def get_set_labels(self):
        return self.set_label

def get_dataset_from_configs(data_cfg, split_idx=None, chunk_size=(10000, 1, 50)):
    """ load miRNA-CTS dataset from config files """
    with open(data_cfg.path[split_idx], "r") as f:
        header = f.readline().strip().split("\t")

    if "split" in header:
        df = dd.read_csv(
            data_cfg.path[split_idx], 
            sep="\t", header=None, 
            usecols=[0,1,2,3,4,5], 
            names=header, 
            skiprows=1
            )
        df = df[df["split"] == split_idx] if split_idx in ["train", "val"] else df
    else:
        df = dd.read_csv(
            data_cfg.path[split_idx], 
            sep="\t", 
            header=None, 
            usecols=[0,1,2,3,4], 
            names=["mirna_id", "mirna_seq", "mrna_id", "mrna_seq", "label"], 
            skiprows=1
            )

    X, labels, set_idxs, set_labels = [], [], [], []
    set_idx = 0


    for i, row in df.iterrows():

        mirna_id, mirna_seq, mrna_id, mrna_seq, label = row[:5]

        mirna_seq = mirna_seq.upper().replace("T", "U")
        mrna_seq = mrna_seq.upper().replace("T", "U")
        mrna_rev_seq = reverse(mrna_seq)

        for pos in range(len(mrna_rev_seq) - 40 + 1):
            mirna_esa, cts_rev_esa, esa_score = extended_seed_alignment(mirna_seq, mrna_rev_seq[pos:pos+40])
            if split_idx not in ["train", "val"] and esa_score < 6: continue
            X.append(encode_RNA(mirna_seq, mirna_esa, mrna_rev_seq[pos:pos+40], cts_rev_esa, data_cfg.with_esa))
            labels.append(np.array(label))
            set_idxs.append(np.array(set_idx))

        set_labels.append(label)
        set_idx += 1

        if set_idx % 5 == 0:
            print('# {} {:.1%}'.format(split_idx, i / len(df)), end='\r', file=sys.stderr)
    print(' ' * 150, end='\r', file=sys.stderr)

    X = da.from_array(X)
    X = X.rechunk(chunk_size)

    labels = da.from_array(labels, chunks=(chunk_size[0],))
    set_idxs = da.from_array(set_idxs, chunks=(chunk_size[0],))
    set_labels = da.from_array(set_labels, chunks=(chunk_size[0],))

    dataset = miRNA_CTS_Dask_Dataset(X, labels, set_idxs, set_labels)

    return dataset


def encode_RNA(mirna_seq, mirna_esa, cts_rev_seq, cts_rev_esa, with_esa):
    """ one-hot encoder for RNA sequences with/without extended seed alignments """
    chars = {"A":0, "C":1, "G":2, "U":3}
    if not with_esa:
        x = np.zeros((len(chars) * 2, 40), dtype=np.float32)
        for i in range(len(mirna_seq)):
            x[chars[mirna_seq[i]], 5 + i] = 1
        for i in range(len(cts_rev_seq)):
            x[chars[cts_rev_seq[i]] + len(chars), i] = 1
    else:
        chars["-"] = 4
        x = np.zeros((len(chars) * 2, 50), dtype=np.float32)
        for i in range(len(mirna_esa)):
            x[chars[mirna_esa[i]], 5 + i] = 1
        for i in range(10, len(mirna_seq)):
            x[chars[mirna_seq[i]], 5 + i - 10 + len(mirna_esa)] = 1
        for i in range(5):
            x[chars[cts_rev_seq[i]] + len(chars), i] = 1
        for i in range(len(cts_rev_esa)):
            x[chars[cts_rev_esa[i]] + len(chars), i + 5] = 1
        for i in range(15, len(cts_rev_seq)):
            x[chars[cts_rev_seq[i]] + len(chars), i + 5 - 15 + len(cts_rev_esa)] = 1

    return x


def reverse(seq):
    """ reverse the given sequence """
    seq_r = ""
    for i in range(len(seq)):
        seq_r += seq[len(seq) - 1 - i]
    return seq_r


score_matrix = {}  # Allow wobble
for c1 in 'ACGU':
    for c2 in 'ACGU':
        if (c1, c2) in [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')]:
            score_matrix[(c1, c2)] = 1
        elif (c1, c2) in [('U', 'G'), ('G', 'U')]:
            score_matrix[(c1, c2)] = 1
        else:
            score_matrix[(c1, c2)] = 0


def extended_seed_alignment(mi_seq, cts_r_seq):
    """ extended seed alignment """
    alignment = pairwise2.align.globaldx(mi_seq[:10], cts_r_seq[5:15], score_matrix, one_alignment_only=True)[0]
    mi_esa = alignment[0]
    cts_r_esa = alignment[1]
    esa_score = alignment[2]
    return mi_esa, cts_r_esa, esa_score

