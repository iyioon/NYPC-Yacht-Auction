import os
import time
import math
import numpy as np
from collections import namedtuple

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from NeuralNet import NeuralNet  # repo base class
from yacht.pytorch.YachtNNet import YachtNNet

# ====== default hyperparams (mirrors style used in A0G Othello) ======


def dotdict(d): return namedtuple("DotDict", d.keys())(*d.values())


defaultArgs = dotdict({
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 10,
    "batch_size": 128,
    "cuda": torch.cuda.is_available(),
    "hidden": 512,
    "nblocks": 6,
    "dropout": 0.2,
})

# ====== dataset for (state, pi, v) ======


class AZDataset(Dataset):
    def __init__(self, xs, pis, vs):
        self.xs = xs.astype(np.float32)
        self.pis = pis.astype(np.float32)
        self.vs = vs.astype(np.float32).reshape(-1, 1)

    def __len__(self): return len(self.xs)

    def __getitem__(self, i):
        return self.xs[i], self.pis[i], self.vs[i]

# ====== state encoder (must match YachtGame.getBoardSize doc) ======
# F = 59: [round/phase 3] + [my 10] + [opp 10] + [rollA 5] + [rollB 5] + [my used 12] + [opp used 12] + [bid 2]


def _scale_die(d):     # 1..6 -> roughly [-0.71, 0.71]; pad -1 -> -1.0
    return -1.0 if d < 1 else (d - 3.5) / 3.5


def _pad_scale(dice, n):
    arr = np.full(n, -1.0, dtype=np.float32)
    for i, v in enumerate(dice[:n]):
        arr[i] = _scale_die(v)
    return arr


def _mask_bits(mask, n):
    return np.array([(mask >> i) & 1 for i in range(n)], dtype=np.float32)


def state_to_vec(game, s):
    # canonical s: p1 = me, p2 = opp
    vec = []
    # round/phase
    vec.append(s.round_no / 13.0)
    vec.append(1.0 if s.phase == 0 else 0.0)  # is_bid
    vec.append(1.0 if s.phase == 1 else 0.0)  # is_score
    # dice
    vec.extend(_pad_scale(s.p1.carry, 10))
    vec.extend(_pad_scale(s.p2.carry, 10))
    # rolls (only present in BID rounds)
    rollA = s.rollA if (s.phase == 0 and s.round_no != 13) else []
    rollB = s.rollB if (s.phase == 0 and s.round_no != 13) else []
    vec.extend(_pad_scale(rollA, 5))
    vec.extend(_pad_scale(rollB, 5))
    # used masks
    vec.extend(_mask_bits(s.p1.used_mask, 12))
    vec.extend(_mask_bits(s.p2.used_mask, 12))
    # bid scores scaled (1e-5 keeps magnitude modest)
    vec.append(s.p1.bid_score * 1e-5)
    vec.append(s.p2.bid_score * 1e-5)
    return np.asarray(vec, dtype=np.float32)

# ====== wrapper ======


class NNetWrapper(NeuralNet):
    def __init__(self, game, args=None):
        super().__init__(game)
        self.game = game
        self.args = args or defaultArgs
        self.input_len = 59
        self.action_size = self.game.getActionSize()

        self.nnet = YachtNNet(
            input_len=self.input_len,
            action_size=self.action_size,
            hidden=self.args.hidden,
            nblocks=self.args.nblocks,
            dropout=self.args.dropout,
        )

        if self.args.cuda:
            self.nnet.cuda()
        self.optimizer = optim.AdamW(self.nnet.parameters(
        ), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Add mixed precision training for better GPU utilization
        if self.args.cuda:
            from torch.amp import GradScaler
            # ---- training on self-play (board, pi, v) ----
            self.scaler = GradScaler('cuda')

    def train(self, examples):
        self.nnet.train()
        xs, pis, vs = [], [], []
        for (board, pi, v) in examples:
            xs.append(state_to_vec(self.game, board))
            pis.append(pi)
            vs.append(v)
        dataset = AZDataset(np.array(xs), np.array(pis), np.array(vs))
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True,
            drop_last=False, num_workers=2, pin_memory=True)  # Added optimizations

        for epoch in range(self.args.epochs):
            total_loss = 0
            batch_count = 0
            for batch_x, batch_pi, batch_v in loader:
                if self.args.cuda:
                    batch_x, batch_pi, batch_v = batch_x.cuda(non_blocking=True), batch_pi.cuda(
                        non_blocking=True), batch_v.cuda(non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                # Use mixed precision if CUDA is available
                if self.args.cuda and hasattr(self, 'scaler'):
                    from torch.amp import autocast
                    with autocast('cuda'):
                        out_pi, out_v = self.nnet(batch_x)
                        loss_policy = F.cross_entropy(
                            out_pi, torch.argmax(batch_pi, dim=1))
                        loss_value = F.mse_loss(out_v, batch_v)
                        loss = loss_policy + self.args.vloss_weight * loss_value

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.nnet.parameters(), max_norm=5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    out_pi, out_v = self.nnet(batch_x)
                    loss_policy = F.cross_entropy(
                        out_pi, torch.argmax(batch_pi, dim=1))
                    loss_value = F.mse_loss(out_v, batch_v)
                    loss = loss_policy + self.args.vloss_weight * loss_value
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.nnet.parameters(), max_norm=5.0)
                    self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            # Print progress every few epochs
            if epoch % 5 == 0 or epoch == self.args.epochs - 1:
                avg_loss = total_loss / batch_count if batch_count > 0 else 0
                print(
                    f"Epoch {epoch+1}/{self.args.epochs}, Avg Loss: {avg_loss:.4f}")

    # ---- inference ----
    def predict(self, board):
        self.nnet.eval()
        x = torch.from_numpy(state_to_vec(
            self.game, board)).unsqueeze(0).float()
        if self.args.cuda:
            x = x.cuda(non_blocking=True)

        with torch.no_grad():
            # Use mixed precision for inference too
            if self.args.cuda:
                from torch.amp import autocast
                with autocast('cuda'):
                    pi, v = self.nnet(x)
            else:
                pi, v = self.nnet(x)

            pi = F.log_softmax(pi, dim=1).exp().cpu().numpy()[0]  # probs
            v = v.cpu().numpy()[0, 0]
        return pi, v

    # ---- checkpoints ----
    def save_checkpoint(self, folder, filename):
        path = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)
        torch.save({
            "state_dict": self.nnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "args": dict(self.args),
        }, path)

    def load_checkpoint(self, folder, filename, load_optimizer=False):
        path = os.path.join(folder, filename)
        checkpoint = torch.load(
            path, map_location="cuda" if self.args.cuda else "cpu", weights_only=False)
        self.nnet.load_state_dict(checkpoint["state_dict"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
