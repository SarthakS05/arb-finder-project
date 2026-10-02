import math
import time
import random
import threading
import socket
import struct
import hashlib
import requests
import numpy as np
import pandas as pd
import cv2
from collections import deque
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class S1BudgetForcingModel(nn.Module):
    def __init__(self, base_model, end_token_id, wait_token_id, max_thinking_tokens=2048):
        super().__init__()
        self.base_model = base_model
        self.end_token_id = end_token_id
        self.wait_token_id = wait_token_id
        self.max_thinking_tokens = max_thinking_tokens

    def forward(self, input_ids, attention_mask=None):
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask)

    def generate_with_budget_forcing(self, input_ids, tokenizer, min_reflections=0, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.max_thinking_tokens
        generated = input_ids.clone()
        thinking_tokens = 0
        reflections = 0
        while True:
            outputs = self.base_model(input_ids=generated)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            token_id = next_token.item()
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            thinking_tokens += 1
            if thinking_tokens >= max_tokens:
                generated = torch.cat(
                    [generated, torch.tensor([[self.end_token_id]])],
                    dim=1
                )
                break
            if token_id == self.end_token_id:
                if reflections < min_reflections:
                    reflections += 1
                    wait_token = torch.tensor([[self.wait_token_id]])
                    generated = torch.cat([generated[:, :-1], wait_token], dim=1)
                    continue
                else:
                    break
        return generated


class BertConfig:
    def __init__(self, vocab_size=30000, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        w = self.word_embeddings(input_ids)
        p = self.position_embeddings(position_ids)
        t = self.token_type_embeddings(token_type_ids)
        x = w + p + t
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, config.hidden_size)
        self.o = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, attention_mask=None):
        b, t, h = x.size()
        q = self.q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(b, t, h)
        out = self.o(ctx)
        return out


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attn = BertSelfAttention(config)
        self.attn_ln = nn.LayerNorm(config.hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.ff_ln = nn.LayerNorm(config.hidden_size)

    def forward(self, x, attention_mask=None):
        a = self.attn(x, attention_mask)
        x = self.attn_ln(x + a)
        f = self.ff(x)
        x = self.ff_ln(x + f)
        return x


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        x = self.embeddings(input_ids, token_type_ids)
        x = self.encoder(x, attention_mask)
        return x


class BertForMaskedLM(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        x = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits, loss


class BertForSequenceClassification(nn.Module):
    def __init__(self, config: BertConfig, num_labels=2):
        super().__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        x = self.bert(input_ids, token_type_ids, attention_mask)
        cls = x[:, 0, :]
        logits = self.classifier(cls)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss


def train_bert_mlm(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits, loss = model(input_ids=input_ids, labels=labels)
        loss.backward()
        optimizer.step()


def fine_tune_bert_classification(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits, loss = model(input_ids=input_ids, labels=labels)
        loss.backward()
        optimizer.step()


def ar_opencv_main():
    cap = cv2.VideoCapture(0)
    pattern_size = (7, 7)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if found:
            cv2.drawChessboardCorners(frame, pattern_size, corners, found)
        cv2.imshow("AR", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


class TorrentFile:
    def __init__(self, info_hash, trackers, length, piece_length, pieces):
        self.info_hash = info_hash
        self.trackers = trackers
        self.length = length
        self.piece_length = piece_length
        self.pieces = pieces


class BitTorrentClient:
    def __init__(self, torrent: TorrentFile, peer_id=None):
        self.torrent = torrent
        self.peer_id = peer_id or self._generate_peer_id()
        self.peers = []
        self.sock = None

    def _generate_peer_id(self):
        s = "-PY0001-"
        r = "".join(str(random.randint(0, 9)) for _ in range(12))
        return (s + r).encode()

    def add_peer(self, ip, port):
        self.peers.append((ip, port))

    def handshake(self, ip, port):
        pstr = b"BitTorrent protocol"
        msg = struct.pack("!B", len(pstr)) + pstr + b"\x00" * 8 + self.torrent.info_hash + self.peer_id
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((ip, port))
        s.sendall(msg)
        resp = s.recv(68)
        self.sock = s
        return resp

    def request_piece(self, index, begin, length):
        if self.sock is None:
            return
        msg = struct.pack("!IbIII", length + 9, 6, index, begin, length)
        self.sock.sendall(msg)


class CryptoMinuteDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_minutes: List[int], horizon=120):
        self.df = df
        self.feature_minutes = feature_minutes
        self.horizon = horizon
        self.indexes = self._build_indexes()

    def _build_indexes(self):
        idx = []
        for t in range(max(self.feature_minutes), len(self.df) - self.horizon - 1):
            idx.append(t)
        return idx

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, i):
        t = self.indexes[i]
        row = self.df.iloc[t]
        prices = self.df["close"].values
        feats = []
        for m in self.feature_minutes:
            p_now = prices[t]
            p_past = prices[t - m]
            r = p_now / p_past - 1.0
            feats.append(r)
        future = prices[t + 1 + self.horizon] / prices[t + 1] - 1.0
        x = np.array(feats, dtype=np.float32)
        y = np.array(future, dtype=np.float32)
        return {"x": x, "y": y}


class CryptoRFStrategy:
    def __init__(self, feature_minutes: List[int]):
        from sklearn.ensemble import RandomForestClassifier
        self.feature_minutes = feature_minutes
        self.model = RandomForestClassifier(n_estimators=1000, max_depth=15)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def backtest_crypto_stat_arb(prices: Dict[str, pd.DataFrame], feature_minutes: List[int], horizon=120, top_k=3, short_k=3, tc_half_turn=0.0015):
    coins = list(prices.keys())
    aligned = []
    for c in coins:
        df = prices[c][["timestamp", "close"]].copy()
        df = df.set_index("timestamp")
        aligned.append(df)
    all_df = pd.concat(aligned, axis=1, keys=coins)
    all_df = all_df.dropna()
    timestamps = all_df.index.values
    returns = all_df.pct_change().fillna(0.0)
    rf = CryptoRFStrategy(feature_minutes)
    X_train = []
    y_train = []
    for t_idx in range(max(feature_minutes), len(timestamps) - horizon - 1):
        feats = []
        for c in coins:
            p_series = all_df[c].values
            p_now = p_series[t_idx]
            for m in feature_minutes:
                p_past = p_series[t_idx - m]
                feats.append(p_now / p_past - 1.0)
        future_ret = []
        for c in coins:
            p_series = all_df[c].values
            r = p_series[t_idx + 1 + horizon] / p_series[t_idx + 1] - 1.0
            future_ret.append(r)
        median_ret = np.median(future_ret)
        labels = [1 if r >= median_ret else 0 for r in future_ret]
        X_train.append(feats)
        y_train.extend(labels)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    rf.fit(X_train, y_train)
    equity = 1.0
    daily_equity = {}
    for t_idx in range(max(feature_minutes), len(timestamps) - horizon - 1):
        feats = []
        for c in coins:
            p_series = all_df[c].values
            p_now = p_series[t_idx]
            for m in feature_minutes:
                p_past = p_series[t_idx - m]
                feats.append(p_now / p_past - 1.0)
        probs = rf.predict_proba(np.array([feats], dtype=np.float32))
        ranks = np.argsort(-probs)
        long_idx = ranks[:top_k]
        short_idx = ranks[-short_k:]
        long_ret = 0.0
        short_ret = 0.0
        for idx in long_idx:
            c = coins[idx % len(coins)]
            p_series = all_df[c].values
            r = p_series[t_idx + 1 + horizon] / p_series[t_idx + 1] - 1.0
            long_ret += r
        for idx in short_idx:
            c = coins[idx % len(coins)]
            p_series = all_df[c].values
            r = p_series[t_idx + 1 + horizon] / p_series[t_idx + 1] - 1.0
            short_ret -= r
        long_ret /= max(1, len(long_idx))
        short_ret /= max(1, len(short_idx))
        gross = (long_ret + short_ret) / 2.0
        net = gross - 2 * tc_half_turn
        equity *= (1.0 + net)
        day = pd.to_datetime(timestamps[t_idx], unit="s").date()
        daily_equity[day] = equity
    return daily_equity


def main():
    pass


if __name__ == "__main__":
    main()

