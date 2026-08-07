# Bitcoin Fraud Detection with Graph Neural Networks

> **Detecting illicit transactions in the Elliptic Bitcoin dataset using GraphSAGE,  
> temporal burst analysis, and graph-aware feature engineering.**
>
> *Opeyemi Mercy Lawan — 2025*

---

## Abstract

Financial crime on blockchain networks presents a unique detection challenge: transaction graphs are large (200k+ nodes), severely imbalanced (< 2% illicit among labelled nodes), and 77% of nodes carry no ground-truth label. Classical machine learning treats nodes independently, discarding the rich structural information encoded in *who transacts with whom*.

This project applies **GraphSAGE** — an inductive graph neural network — to the Elliptic Bitcoin dataset, augmented with three families of hand-engineered graph features (structural centrality, temporal burst signals, and neighbourhood contagion scores). A temporal train/test split (steps 1–34 train, 35–49 test) ensures evaluation mirrors real-world deployment conditions.

Key contributions:
1. **Graph-aware feature engineering** — 16 new features across three families added to the original 165-dimensional Elliptic feature vector
2. **GraphSAGE classifier** — 3-layer GNN with residual connections, BatchNorm, and class-weighted loss achieving AUC 0.957 and Recall 0.91
3. **Unknown node labelling** — fraud probabilities assigned to 157,205 previously-unclassified wallets, identifying 88,234 as high-risk — a **1,971% increase** in detectable suspicious activity
4. **Temporal burst detection** — z-score anomaly flagging identifies 6 statistically significant burst windows (steps 9, 13, 20, 28, 29, 32)
5. **Propagation analysis** — illicit activity spreads to 35.57% of exposed nodes on average, peaking at 84% at time step 29
6. **Interactive risk dashboard** — self-contained HTML with model gauges, network graph, burst chart, ranking table, and timeline
7. **Compliance report** — non-technical stakeholder report covering highest-risk nodes, suspicious clusters, and actionable recommendations

---

## Results

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.957** |
| Recall (Illicit) | **0.910** |
| F1 Score (Illicit) | **0.424** |
| Average Precision | **0.838** |
| Confirmed illicit nodes | 4,545 |
| GNN-predicted illicit (unknown) | 88,234 |
| Total at-risk nodes | 92,779 |
| Detection increase | **1,971%** |
| Burst time steps (z > 1.5σ) | 9, 13, 20, 28, 29, 32 |
| Mean propagation rate | 35.57% |
| Peak propagation rate | 84.11% at t=29 |
| Early warning MAE | 0.058 |

---

## Project Structure

```
bitcoin-fraud-gnn/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_EDA.ipynb                 ← Class imbalance, temporal distribution,
│   │                                   feature analysis, graph topology
│   ├── 02_Feature_Engineering.ipynb ← Structural, temporal & neighbourhood features
│   ├── 03_GNN_Classifier.ipynb      ← GraphSAGE model, training, evaluation,
│   │                                   unknown node predictions, t-SNE embeddings
│   └── 04_Temporal_Analysis.ipynb   ← Burst detection, propagation analysis,
│                                       early warning model, combined timeline
│
├── src/
│   ├── graph_builder.py             ← Graph loading and construction utilities
│   ├── features.py                  ← FeatureBuilder class (all 3 feature families)
│   └── model.py                     ← GraphSAGE model + Trainer class
│
├── data/
│   └── README.md                    ← Download instructions for Elliptic dataset
│
├── dashboard/
│   └── Bitcoin_Network_Risk_Intelligence_Dashboard.html
│
└── report/
    └── Compliance_Report_Bitcoin_Network.docx
```

---

## Quickstart (Google Colab)

All notebooks are designed to run on **Google Colab**. Notebooks 01, 02 and 04 run on free CPU. Notebook 03 benefits from a T4 GPU (Runtime → Change runtime type → T4 GPU).

### Step 1 — Get the data
```python
# Install Kaggle API
!pip install kaggle -q

# Set up credentials
import os
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    f.write('{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}')
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

# Download dataset
!kaggle datasets download -d ellipticco/elliptic-data-set

# Unzip
import zipfile
with zipfile.ZipFile('elliptic-data-set.zip', 'r') as z:
    z.extractall('data')
```

### Step 2 — Run notebooks in order
```
01_EDA.ipynb → 02_Feature_Engineering.ipynb → 03_GNN_Classifier.ipynb → 04_Temporal_Analysis.ipynb
```

### Local setup
```bash
git clone https://github.com/YOUR_USERNAME/bitcoin-fraud-gnn.git
cd bitcoin-fraud-gnn
pip install -r requirements.txt
jupyter notebook
```

---

## Methodology

### Dataset
The Elliptic Bitcoin Dataset contains 203,769 transaction nodes and 234,355 directed payment edges.

| Class | Count | % of total |
|-------|-------|------------|
| Unknown | 157,205 | 77.1% |
| Licit | 42,019 | 20.6% |
| Illicit | 4,545 | 2.2% |

### Feature Engineering
Three feature families engineered on top of the original 166-dimensional vectors:

| Family | Features | Count |
|--------|----------|-------|
| Structural | In/out degree, betweenness, PageRank, HITS hub/authority, degree ratio | 8 |
| Temporal | Time-step volume, illicit rate at same step, recency | 3 |
| Neighbourhood | Illicit/licit/unknown neighbour ratios, total neighbours, predecessor illicit ratio | 5 |

**Total: 181 features per node**

### Model Architecture
GraphSAGE with:
- 3 SAGEConv layers (mean aggregation)
- Hidden dimension: 128
- BatchNorm after each layer
- Dropout: 0.4
- Residual skip connection (layer 2 → layer 3)
- Weighted cross-entropy loss (7.63× weight on illicit class)
- 113,282 trainable parameters

### Training
- **Optimiser:** Adam (lr=1e-3, weight decay=5e-4)
- **Scheduler:** ReduceLROnPlateau (patience=10, factor=0.5)
- **Epochs:** 200
- **Split:** Temporal — train on steps 1–34, test on 35–49

### Key Findings

**1. Fraud is structurally organised**
Illicit nodes show near-perfect feature correlation — consistent with automated fraud scripts or coordinated laundering operations.

**2. Fraud clusters temporally**
Six burst windows identified (steps 9, 13, 20, 28, 29, 32). Step 13 shows the highest illicit rate at 35%.

**3. Fraud propagates deliberately**
35.57% of nodes that receive funds from illicit nodes become illicit themselves at the next time step — 3.5× the base rate. Peak propagation of 84% at step 29 is consistent with layering in money laundering.

**4. GNN dramatically expands detection**
88,234 previously-unknown wallets identified as high-risk — a 1,971% increase in detectable suspicious activity.

---

## Dashboard

The interactive dashboard is a self-contained HTML file requiring no server:

- **Model performance gauges** — AUC, Recall, F1, Average Precision
- **Interactive network graph** — 500 highest-risk nodes, coloured by class, top-10 in gold
- **Node detail panel** — click any node to see its full risk profile
- **Ranking table** — top-20 nodes by GNN risk score
- **Combined timeline** — confirmed + predicted illicit by time step
- **Burst detection chart** — illicit rate with statistical anomaly flagging
- **Class filter** — filters all components simultaneously
- **Footer** — project attribution and key metrics

---

## References

1. Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.
2. Weber, M. et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*. KDD Workshop on Anomaly Detection in Finance.
3. Elliptic (2019). *Elliptic Bitcoin Dataset*. Kaggle. https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
4. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.

---

## Author

**Opeyemi Mercy Lawan**  
Bitcoin Transaction Network · Fraud Detection Research · 2025

*This project was completed as part of a fraud analytics internship and extended into independent research.*
