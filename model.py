import os
import pickle
from dataclasses import dataclass
from typing import List, Callable

import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab

@dataclass
class Config:
    vocab_size: int = 8100
    max_seq_len: int = 30
    d_model: int = 256
    n_heads: int = 4
    depth: int = 1
    n_classes: int = 4
    dropout: float = 0.1
    batch_size: int = 128
    max_epochs: int = 20

class TextEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding.from_pretrained(
            embeddings=self.get_sinusoidal_encoding(config.max_seq_len, config.d_model),
            freeze=True)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = 0.1
    
    def forward(self, text_ids: torch.Tensor):
        """Embed content and positions of tokens.
        Args:
            text_ids: Tensor, shape[bsz, seq_len]
        """
        pos = torch.arange(text_ids.size(1)).unsqueeze(0).type_as(text_ids)
        embedding = self.text_embedding(text_ids)
        embedding += self.pos_embedding(pos)

        embedding = self.layer_norm(embedding)
        embedding = F.dropout(embedding, self.dropout, self.training)
        return embedding

    def get_sinusoidal_encoding(self, max_position_embeddings, embedding_dim):
        position_embedding = torch.zeros(max_position_embeddings, embedding_dim)
        positions = torch.arange(end=max_position_embeddings).unsqueeze(1)
        angle_denominators = 10000.**(torch.arange(0, embedding_dim, 2)/embedding_dim)
        angles = positions/angle_denominators
        position_embedding[:, ::2] = torch.sin(angles)
        position_embedding[:, 1::2] = torch.cos(angles)
        return position_embedding

class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = TextEmbedding(config)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, config.depth)
        self.predict_head = self.build_predict_head()

        self.save_hyperparameters()

    def forward(self, text_ids):
        """
        Args:
            text_ids: Tensor, shape[bsz, seq_len]
        """
        mask = (text_ids!=0).int()
        outputs = self.embedding(text_ids)
        outputs = self.transformer_encoder(outputs, src_key_padding_mask=(1-mask).bool())
        outputs = self.average_readout(outputs, mask)
        outputs = self.predict_head(outputs)
        if not self.training:
            outputs = F.softmax(outputs, dim=-1)
        return outputs

    def average_readout(self, hidden, mask):
        """
        Args:
            hidden: Tensor, shape[bsz, seq_len, d_model]
            mask: Tensor, shape[bsz, seq_len]. Padded elements are labeled 0.
        """
        unsqueeze_mask = mask.unsqueeze(1).float()
        return torch.bmm(unsqueeze_mask, hidden).squeeze(1)/unsqueeze_mask.sum(-1)
    
    def build_predict_head(self):
        inter_dim = int(0.5*self.config.d_model)
        return nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model, inter_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(inter_dim, self.config.n_classes),
        )
    
    def training_step(self, batch_item, batch_idx):
        inputs, true_labels = batch_item
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, true_labels, reduction="mean")
        return loss
    
    def training_epoch_end(self, train_outputs):
        print(f'Epoch {self.current_epoch}: train_loss = {train_outputs[-1]["loss"]}')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def fit_vocab(texts: List[str], tokenizer):
    vocab = build_vocab_from_iterator(map(tokenizer, texts), min_freq=2, specials=["<pad>"])
    vocab.set_default_index(len(vocab)) #OOV token
    return vocab

label2id = {"b":0, "e":1, "m":2, "t":3}
id2label = {v:k for k, v in label2id.items()}

def get_text2tensor_fn(config, 
                       tokenizer, 
                       vocab: Vocab):
    def text2tensor(text: str):
        """Turn text_ids to tensor and pad to a max len
        """
        text_tensor = torch.tensor(vocab(tokenizer(text)), dtype=torch.int64)
        text_tensor = F.pad(text_tensor, (0, config.max_seq_len-text_tensor.size(0)))
        return text_tensor.unsqueeze(0)
    
    return text2tensor

class MyDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
    def __len__(self):
        return len(self.df)
    def __getitem__(self, id):
        return self.df.iloc[id, 0], self.df.iloc[id, 1]

def get_collate_fn(text2tensor: Callable):

    def collate_fn(batch_item):
        text_tensors = []
        labels = []
        for text, label in batch_item:
            text_tensor = text2tensor(text)
            text_tensors.append(text_tensor)
            labels.append(label2id[label])
        text_tensors = torch.cat(text_tensors, )
        labels = torch.tensor(labels, dtype=torch.int64)
        return text_tensors, labels
    
    return collate_fn

class TextPreprocessing:
    def __init__(
        self,
        max_seq_len,
        train_df=None,
        vocab_save_file=None,
        ):
        self.tokenizer = get_tokenizer("basic_english")
        if vocab_save_file is None or not os.path.exists(vocab_save_file):
            if train_df is None:
                raise ValueError("train_df and vocab_save_file cannot be both None")
            else:
                self.vocab = self.fit_vocab(train_df.loc[:, "TITLE"], self.tokenizer)
                with open(vocab_save_file, "wb") as file:
                    torch.save(vocab, file)
        else:
            self.vocab = torch.load(vocab_save_file)
        self.max_seq_len = max_seq_len
    
    def fit_vocab(self, texts: List[str], tokenizer):
        vocab = build_vocab_from_iterator(map(tokenizer, texts), min_freq=2, specials=["<pad>"])
        vocab.set_default_index(len(vocab)) #OOV token
        return vocab

    def text2tensor(self, text: str):
        """Turn text_ids to tensor and pad to a max len
        """
        text_tensor = torch.tensor(self.vocab(self.tokenizer(text)), dtype=torch.int64)
        text_tensor = F.pad(text_tensor, (0, self.max_seq_len-text_tensor.size(0)))
        return text_tensor.unsqueeze(0)