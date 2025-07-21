import numpy as np
import matplotlib.pyplot as plt
import flwr as fl
from sklearn.ensemble import IsolationForest

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
NUM_CLIENTS       = 500
ROUNDS            = 100
CLIENTS_PER_ROUND = 100
REP_DECAY         = 0.9
REP_THRESHOLD     = 0.5
EPSILON           = 1e-6
SERVER_LR         = 1.0  # 

# ─── AGGREGATORS ────────────────────────────────────────────────────────────────
def fedavg(updates):
    return [np.mean(np.stack(layer, axis=0), axis=0)
            for layer in zip(*updates)]

def krum(updates, f=1):
    n = len(updates)
    scores = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j: continue
            dists.append(sum(np.linalg.norm(a - b)**2
                             for a, b in zip(updates[i], updates[j])))
        dists.sort()
        scores.append(sum(dists[: n - f - 1]))
    return updates[int(np.argmin(scores))]

def trimmed_mean(updates, beta=1):
    result = []
    for layer in zip(*updates):
        arr = np.stack(layer, axis=0)
        arr_sorted = np.sort(arr, axis=0)
        trimmed = arr_sorted[beta:-beta] if beta > 0 else arr_sorted
        result.append(np.mean(trimmed, axis=0))
    return result

# ─── FLIFRA STRATEGY ────────────────────────────────────────────────
class FLIFRAStrategy(fl.server.strategy.Strategy):
    def __init__(self):
        self.reps = np.ones(NUM_CLIENTS)
        self.global_weights = None

    def initialize_parameters(self, client_manager):
        return None  # let first client seed

    def configure_fit(self, rnd, parameters, client_manager):
        clients = client_manager.sample(
            num_clients=CLIENTS_PER_ROUND,
            min_num_clients=CLIENTS_PER_ROUND
        )
        return clients, parameters, {}

    def aggregate_fit(self, rnd, results, failures):
        # updates
        updates = [r.parameters for r in results]
        #  median reference
        stacked = [np.stack([u[layer] for u in updates], axis=0)
                   for layer in range(len(updates[0]))]
        U_med = [np.median(s, axis=0) for s in stacked]
        #  distances
        dists = [np.sqrt(sum(np.linalg.norm(u_l - m_l)**2
                             for u_l, m_l in zip(u, U_med)))
                 for u in updates]
        D_max = max(max(dists), EPSILON)
        # scores & reputations
        weights = []
        for i, d in enumerate(dists):
            score = max(0, min(1, 1 - d / D_max))
            self.reps[i] = REP_DECAY * self.reps[i] + (1 - REP_DECAY) * score
            w = self.reps[i] if self.reps[i] >= REP_THRESHOLD else 0
            weights.append(w)
        w_arr = np.array(weights)
        if w_arr.sum() == 0:
            w_arr = np.ones_like(w_arr)
        w_arr /= w_arr.sum()
        # weighted aggregate
        new_global = [np.tensordot(w_arr, stacked[layer], axes=(0, 0))
                      for layer in range(len(stacked))]
        # Algorithm 3: update global model
        if self.global_weights is None:
            self.global_weights = new_global
        else:
            self.global_weights = [
                old + SERVER_LR * (new - old)
                for old, new in zip(self.global_weights, new_global)
            ]
        return self.global_weights, {}

    def configure_evaluate(self, rnd, parameters, client_manager):
        clients = client_manager.sample(
            num_clients=NUM_CLIENTS,
            min_num_clients=NUM_CLIENTS
        )
        return clients, parameters, {}

    def aggregate_evaluate(self, rnd, results, failures):
        accs = [r.metrics["accuracy"] for r in results]
        return float(np.mean(accs)), {}

# ─── RUN──────────────────────────────────────────
if __name__ == "__main__":
    # Compare also FedAvg, Krum, Trimmed as baselines
    strategies = {
        "FedAvg": fl.server.strategy.FedAvg(),
        "Krum": fl.server.strategy.AggregateStrategy(
            aggregate_fit_fn=lambda r, res, f: (krum([x.parameters for x in res]), {}),
            evaluate_fn=None
        ),
        "Trimmed": fl.server.strategy.AggregateStrategy(
            aggregate_fit_fn=lambda r, res, f: (trimmed_mean([x.parameters for x in res]), {}),
            evaluate_fn=None
        ),
        "FLIFRA": FLIFRAStrategy(),
    }

    histories = {}
    for name, strat in strategies.items():
        hist = fl.simulation.start_simulation(
            client_fn=lambda cid: FLIFRAClient(cid),
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=ROUNDS),
            strategy=strat,
        )
        histories[name] = hist.metrics_centralized["accuracy"]

    # Plot comparison
    plt.figure(figsize=(8, 6))
    for name, acc in histories.items():
        plt.plot(range(1, ROUNDS+1), acc, marker="o", label=name)
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cicids_2018_10.eps", format="eps")
    plt.show()
