# FLIFRA: Hybrid Data Poisoning Attack Detection in Federated Learning for IoT Security

This repository contains the implementation of **FLIFRA** (Federated Learning with iForest anomaly filtering and Dynamic Reputation–Based Robust Aggregation) that a dual-layer approach for client and server-side data to pose attack detection in the NIDS model.

**Key Components:**

* **Client (`client.py`)**: Implements

  * Data loading & non‑IID Dirichlet split of 100 clients
  * adversarial clients flipping their labels
  * adding noise on the client side 
  * iForest anomaly scoring, detection, filtering
  * Local training of the model
  * uploading to the server
* **Server (`server.py`)**: Implements
  * flower-federted learning settings 
  * Baselines: FedAvg, Krum, Trimmed‑Mean, DRRA, WeiDetect
  * DRRA reputation‑weighted aggregation
  * FLIFRA hybrid (iForest + DRRA)
  * Simulation loop over 100 communication rounds
  * Test accuracy
  * Flower federted learning that simulates for 100 different clients 100 clients
  * Different posisning intensites (10%, 20% , 30%, 40%)

## 📋 Features

* **data preprocessing (cic-ids2018, UNSW-NB15 and BOT-IoT datasest)
* **Non‑IID data** via Dirichlet($\alpha=0.5$) splitting across 100 clients
* **Poisoning simulation**: 10%, 20%, 30% and 40% of clients adversarially flip their labels and noise
* **Anomaly filtering**: iForest (100 trees, n% contamination)
* **Robust aggregation**: Dynamic Reputation–Based Robust Aggregation (DRRA)
* **Baselines**: FedAvg, Krum, Trimmed‑Mean, DRRA, WeiDetect for comparative evaluation
## 🚀 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/mulerkal/flifra
   cd flifra
  
2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
 

4. **Prepare dataset**:

   * Place the dataset in `data.csv`.
   * Ensure it has a target column named `Label` (string labels).
   * Modify `DATA_PATH` and `LABEL_COLUMN` in both scripts if needed.
  
## 🔍 Sensitivity Analysis

We conducted a sensitivity analysis of our dual‑layer poisoning defense by varying the iForest contamination parameter η over the set {0.01, 0.02, 0.05}. For each η, we ran federated learning experiments under consistent poisoning scenarios and recorded:

* **True Positive Rate (TPR)**: fraction of poisoned samples correctly filtered
* **False Alarm Rate (FAR)**: fraction of benign samples incorrectly filtered


## ⚙️ Configuration

All hyperparameters and experiment settings are defined at the top of each script:

* `NUM_CLIENTS`, `NUM_CLASSES`, `ROUNDS`, `CLIENTS_PER_ROUND`
* Poisoning parameters: `POISON_CLIENT_FRAC`, `POISON_LABEL_FRAC`
* Dirichlet split: `DIRICHLET_ALPHA`
* Isolation Forest: `IF_TREES`, `CONTAM_IF`
* DRRA: `REP_DECAY`, `REP_THRESHOLD`, `SERVER_LR`

## 🎯 Usage

1. **Start the server**:

   ```bash
   python server.py
   ```

2. **Start a client** (in a separate terminal):

   ```bash
   python client.py
   ```

3. **Run multiple clients**: In practice, spawn multiple instances (up to 100) of `python client.py`. Each will simulate one FLIFRAClient.

4. **View results**:

   * After simulation, the server will save `cicids_10percen.eps` with accuracy curves.
   * You can convert or view this EPS in your favorite plotting tool.

## 📖 Citation

If you use this code, please cite our paper:

> **Anley, Mulualem Bitew and Genovese, Angelo, Tesema, Tibebe Beshah and Piuri, Vincenzo, “FLIFRA: Hybrid Data Poisoning Attack Detection
In Federated Learning for IoT Security, *IEEE SMC conference *, 2025.

```bibtex
@inproceedings{your2025flifra,
  author    = {{Anley, Mulualem Bitew and Genovese, Angelo, Tesema, Tibebe Beshah and Piuri, Vincenzo}},
  title     = {FLIFRA: Hybrid Data Poisoning Attack Detection
In Federated Learning for IoT Security,
  booktitle = {IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  year      = {2025},
}
```

Access to this codebase is provided under the condition that you acknowledge and cite the above paper in any derivative work.

---

## 📝 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

