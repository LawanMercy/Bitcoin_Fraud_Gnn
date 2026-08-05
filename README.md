# Bitcoin Fraud Detection with Graph Neural Networks

> **Detecting illicit transactions in the Elliptic Bitcoin dataset using GraphSAGE,  
> temporal burst analysis, and graph-aware feature engineering.**

---

## Abstract

Financial crime on blockchain networks presents a unique detection challenge: transaction graphs are large (200k+ nodes), severely imbalanced (< 2% illicit), and 77% of nodes carry no ground-truth label. Classical machine learning treats nodes independently, discarding the rich structural information encoded in *who transacts with whom*.

This project applies **GraphSAGE** — an inductive graph neural network — to the Elliptic Bitcoin dataset, augmented with three families of hand-engineered graph features (structural centrality, temporal burst signals, and neighbourhood contagion scores). A temporal train/test split (steps 1–34 train, 35–49 test) ensures evaluation mirrors real-world deployment conditions.

Key contributions:
1. **GNN classifier** that learns from both node features and graph topology, achieving competitive F1 on the illicit class
2. **Temporal burst detection** identifying time steps where illicit activity is statistically anomalous (z > 1.5σ)
3. **Propagation analysis** quantifying how illicit activity spreads across time steps
4. **Unknown node labelling** — predictions on 157,205 previously-unclassified wallets, expanding the identifiable illicit pool
5. **Interactive risk dashboard** and compliance report for non-technical stakeholders

---

## Results Summary

| Metric | Value |
|--------|-------|
| F1 Score (Illicit class) | *See notebook 03* |
| AUC-ROC | *See notebook 03* |
| Avg Precision | *See notebook 03* |
| Illicit bursts detected | Time steps with z > 1.5σ |
| Previously-unknown wallets labelled | 157,205 |

*Results vary with random seed and hardware. Run the notebooks to reproduce.*

---

## Project Structure

```
bitcoin-fraud-gnn/
│
├── README.md                        ← This file
├── requirements.txt
│
├── notebooks/
│   ├── 01_EDA.ipynb                 ← Dataset exploration, class imbalance, topology
│   ├── 02_Feature_Engineering.ipynb ← Structural, temporal & neighbourhood features
│   ├── 03_GNN_Classifier.ipynb      ← GraphSAGE model, training, evaluation, t-SNE
│   ├── 04_Temporal_Analysis.ipynb   ← Burst detection, propagation, early warning
│   └── 05_Dashboard_Export.ipynb    ← Interactive HTML risk dashboard
│
├── src/
│   ├── graph_builder.py             ← Graph loading and construction utilities
│   ├── features.py                  ← FeatureBuilder class (all 3 feature families)
│   └── model.py                     ← GraphSAGE model + Trainer class
│
├── data/
│   ├── README.md                    ← Download instructions
│   └── [generated files]            ← Created when you run the notebooks
│
├── dashboard/
│   └── risk_dashboard.html          ← Self-contained interactive dashboard
│
└── report/
    └── compliance_report.docx       ← Non-technical stakeholder report
```

---

## Quickstart (Google Colab)

All notebooks are designed to run on **Google Colab** (free tier is sufficient for notebooks 01–02 and 04; notebook 03 benefits from a GPU runtime).

```
1. Download the Elliptic dataset from Kaggle → place in data/
2. Open notebooks in order: 01 → 02 → 03 → 04 → 05
3. Notebook 03 installs PyTorch and PyG automatically
```

### Local setup

```bash
git clone https://github.com/YOUR_USERNAME/bitcoin-fraud-gnn.git
cd bitcoin-fraud-gnn
pip install -r requirements.txt
# PyTorch + PyG: follow https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
jupyter notebook
```

---

## Methodology

### Feature Engineering
Three feature families are computed on top of Elliptic's original 166-dimensional node vectors:

| Family | Features | Motivation |
|--------|----------|------------|
| Structural | In/out degree, betweenness, PageRank, HITS hub/authority | Node's influence and position in the network |
| Temporal | Time-step volume, illicit rate at same step, recency | Illicit nodes cluster in time bursts |
| Neighbourhood | Illicit/licit/unknown neighbour ratios, predecessor illicit ratio | Guilt-by-association is empirically validated |

### Model Architecture
GraphSAGE with 3 convolutional layers, mean aggregation, BatchNorm, dropout (p=0.4), and a residual skip connection. Class imbalance is addressed via weighted cross-entropy (~10× weight on the illicit class).

### Evaluation
- **Temporal split**: training on time steps 1–34 prevents data leakage from future transactions
- **Primary metric**: F1 on the illicit class (not accuracy — misleading on imbalanced data)
- **Secondary metrics**: AUC-ROC, Average Precision, Precision-Recall curve

---

## References

1. Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.
2. Weber, M. et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*. KDD Workshop on Anomaly Detection in Finance.
3. Elliptic (2019). *Elliptic Bitcoin Dataset*. Kaggle. https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
4. https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
5. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.

---

## Acknowledgements

Dataset provided by Elliptic (https://www.elliptic.co). This project was completed as part of a personal project and extended into independent research.

---

*Author: [Opeyemi Mercy lawan] · [2026]*
