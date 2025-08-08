"""
PyTorch neural network for the yacht dice game suitable for use with
alpha-zero-general.

This module defines a wrapper class around a PyTorch model.  It converts
between the alpha-zero-general game interface and the network, handles
loading/saving weights and performs inference.  The network architecture
itself is kept deliberately simple; you can extend it to deeper residual
blocks as needed.  See the report for details on designing the input and
output sizes.
"""

from __future__ import annotations

import os
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from yacht.Game import Game


class Net(nn.Module):
    def __init__(self, game: Game):
        super().__init__()
        # Retrieve board and action sizes from the game
        self.board_size = game.getBoardSize()
        self.action_size = game.getActionSize()
        # Flatten input for a fully connected network
        input_dim = self.board_size[0] * \
            self.board_size[1] * self.board_size[2]
        hidden_dim = 256
        # A simple three‑layer MLP; feel free to increase depth and width
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Policy head
        self.policy_head = nn.Linear(hidden_dim, self.action_size)
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x has shape (batch_size, vector_length) for 1D input
        # Ensure input is properly flattened
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Policy: logits over actions
        policy = self.policy_head(x)
        # Value: scalar output with tanh activation
        value = torch.tanh(self.value_head(x))
        return policy, value.squeeze(-1)


class NNetWrapper:
    def __init__(self, game: Game) -> None:
        self.nnet = Net(game)
        self.game = game
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)
        # Optimiser parameters (tune as needed)
        self.lr = 0.001
        self.batch_size = 64
        self.loss_fn = nn.MSELoss()
        self.optimiser = optim.Adam(self.nnet.parameters(), lr=self.lr)

    def train(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        """
        Train the network with a batch of examples.  Each example is a
        tuple (board, pi, v) where `board` is a state representation,
        `pi` is the target policy vector and `v` is the target value.
        """
        self.nnet.train()
        for epoch in range(10):  # number of epochs per training call
            batch_index = 0
            while batch_index < len(examples):
                sample = examples[batch_index: batch_index + self.batch_size]
                boards, pis, vs = zip(*sample)
                # Convert boards (dict) to flat vectors
                boards_vec = [self.game._board_to_vector(b) for b in boards]
                boards = torch.tensor(
                    np.array(boards_vec), dtype=torch.float32, device=self.device)
                pis = torch.tensor(
                    np.array(pis), dtype=torch.float32, device=self.device)
                vs = torch.tensor(
                    np.array(vs), dtype=torch.float32, device=self.device)
                # Forward pass
                out_policy, out_value = self.nnet(boards)
                # Mask invalid moves: ensure probabilities sum to 1 over valid moves
                # It is the caller's responsibility to zero out invalid entries in `pis`.
                policy_loss = - \
                    torch.mean(
                        torch.sum(pis * torch.log_softmax(out_policy, dim=1), dim=1))
                value_loss = self.loss_fn(out_value, vs)
                loss = policy_loss + value_loss
                # Backward and optimise
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                batch_index += self.batch_size

    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Given a board, predict the policy (action probabilities) and value.
        Return a tuple (policy, value) where `policy` is a 1D numpy array
        of length equal to the action size and `value` is a float in
        [−1,1].  The caller must mask invalid moves using
        `game.getValidMoves()` before acting on the policy.
        """
        self.nnet.eval()
        with torch.no_grad():
            # Convert board dict to vector
            board_vec = self.game._board_to_vector(board)
            board_tensor = torch.tensor(
                board_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            policy, value = self.nnet(board_tensor)
            policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
            value = value.item()
        return policy, value

    def save_checkpoint(self, folder: str, filename: str) -> None:
        """Save the network parameters to a file."""
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder: str, filename: str) -> None:
        """Load network parameters from a file."""
        filepath = os.path.join(folder, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
