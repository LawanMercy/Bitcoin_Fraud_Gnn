"""
model.py
--------
GraphSAGE model definition and training utilities.

References
----------
Hamilton et al. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
Weber et al. (2019). Anti-Money Laundering in Bitcoin. KDD Workshop.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    f1_score, roc_auc_score,
    average_precision_score, classification_report
)


class GraphSAGE(nn.Module):
    """
    3-layer GraphSAGE classifier with:
      - Mean neighbourhood aggregation
      - BatchNorm after each layer
      - Dropout regularisation
      - Residual connection (layer 2 → layer 3)

    Parameters
    ----------
    in_channels     : input feature dimension
    hidden_channels : size of hidden representations
    out_channels    : number of classes (2 for binary fraud detection)
    dropout         : dropout probability (default 0.4)
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.4):
        super().__init__()
        self.conv1 = SAGEConv(in_channels,     hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.bn1   = nn.BatchNorm1d(hidden_channels)
        self.bn2   = nn.BatchNorm1d(hidden_channels)
        self.bn3   = nn.BatchNorm1d(hidden_channels)
        self.lin   = nn.Linear(hidden_channels, out_channels)
        self.drop  = dropout

    def forward(self, x, edge_index):
        x  = F.dropout(F.relu(self.bn1(self.conv1(x, edge_index))),
                        p=self.drop, training=self.training)
        x2 = F.dropout(F.relu(self.bn2(self.conv2(x, edge_index))),
                        p=self.drop, training=self.training)
        x3 = F.dropout(F.relu(self.bn3(self.conv3(x2, edge_index) + x2)),
                        p=self.drop, training=self.training)
        return self.lin(x3)

    def embed(self, x, edge_index):
        """Return penultimate-layer embeddings (for t-SNE / visualisation)."""
        self.eval()
        with torch.no_grad():
            x  = F.relu(self.bn1(self.conv1(x, edge_index)))
            x2 = F.relu(self.bn2(self.conv2(x, edge_index)))
            x3 = F.relu(self.bn3(self.conv3(x2, edge_index) + x2))
        return x3

    def predict_proba(self, x, edge_index):
        """Return softmax probabilities for all nodes."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return F.softmax(logits, dim=1)


class Trainer:
    """
    Encapsulates the training loop, evaluation, and model persistence.

    Parameters
    ----------
    model     : GraphSAGE instance
    data      : torch_geometric.data.Data object
    lr        : learning rate (default 1e-3)
    weight_decay : L2 regularisation (default 5e-4)
    """

    def __init__(self, model, data, lr=1e-3, weight_decay=5e-4):
        self.model  = model
        self.data   = data
        self.device = next(model.parameters()).device

        # Class weights from training set
        y_tr = data.y[data.train_mask]
        n_licit   = int((y_tr == 0).sum())
        n_illicit = int((y_tr == 1).sum())
        w = torch.tensor([1.0, n_licit / max(n_illicit, 1)], dtype=torch.float).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=w)
        self.optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, patience=10, factor=0.5, verbose=False
        )
        self.history   = {"train_loss": [], "val_f1": [], "val_auc": []}
        self.best_f1   = 0.0
        self.best_state = None

    def train(self, epochs=200, verbose=True):
        """Run the full training loop."""
        for epoch in range(1, epochs + 1):
            loss = self._train_step()
            f1, auc = self._evaluate(self.data.test_mask)
            self.scheduler.step(1 - f1)
            self.history["train_loss"].append(loss)
            self.history["val_f1"].append(f1)
            self.history["val_auc"].append(auc)

            if f1 > self.best_f1:
                self.best_f1    = f1
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        self.model.load_state_dict(self.best_state)
        if verbose:
            print(f"\n✅ Training complete. Best F1: {self.best_f1:.4f}")
        return self.history

    def evaluate_full(self):
        """Print full classification report on the test set."""
        self.model.eval()
        with torch.no_grad():
            out   = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()

        mask = self.data.test_mask.cpu().numpy()
        y    = self.data.y.cpu().numpy()

        print(classification_report(y[mask], preds[mask],
                                    target_names=["Licit", "Illicit"],
                                    zero_division=0))
        auc = roc_auc_score(y[mask], probs[mask])
        ap  = average_precision_score(y[mask], probs[mask])
        print(f"AUC-ROC: {auc:.4f}  |  Avg Precision: {ap:.4f}")
        return {"auc": auc, "ap": ap}

    def save(self, path: str):
        torch.save(self.best_state, path)
        print(f"Model saved → {path}")

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

    # ── Private ────────────────────────────────────────────────────────────────

    def _train_step(self):
        self.model.train()
        self.optimiser.zero_grad()
        out  = self.model(self.data.x, self.data.edge_index)
        loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimiser.step()
        return loss.item()

    def _evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            out   = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()

        y     = self.data.y[mask].cpu().numpy()
        p     = preds[mask.cpu().numpy()]
        prob  = probs[mask.cpu().numpy()]

        if len(np.unique(y)) < 2:
            return 0.0, 0.0
        f1  = f1_score(y, p, pos_label=1, zero_division=0)
        auc = roc_auc_score(y, prob)
        return f1, auc
