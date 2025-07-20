# server.py

import numpy as np
import matplotlib.pyplot as plt
import flwr as fl

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
NUM_CLIENTS       = 100
ROUNDS            = 50
CLIENTS_PER_ROUND = 20
SERVER_LR         = 1.0
REP_DECAY         = 0.9
REP_THRESHOLD     = 0.5
EPSILON           = 1e-6

# ─── 5) BASELINE AGGREGATORS ────────────────────────────────────────────────────
def fedavg(weights_list):
    return [np.mean(np.stack(l,0),0) for l in zip(*weights_list)]

def krum(weights_list, f= int(0.10*NUM_CLIENTS)):
    n = len(weights_list); scores=[]
    for i in range(n):
        dists=[]
        for j in range(n):
            if i==j: continue
            dists.append(sum(np.sum((a-b)**2) for a,b in zip(weights_list[i], weights_list[j])))
        dists.sort(); scores.append(sum(dists[:n-f-1]))
    return weights_list[int(np.argmin(scores))]

def trimmed_mean(weights_list, beta=int(0.10*NUM_CLIENTS)):
    out=[]
    for layer in zip(*weights_list):
        arr=np.stack(layer,0); s=np.sort(arr,0)
        trimmed = s[beta:-beta] if beta>0 else s
        out.append(np.mean(trimmed,0))
    return out

# ─── 6) FLIFRA STRATEGY (Alg. 2+3) ───────────────────────────────────────────────
class FLIFRAStrategy(fl.server.strategy.Strategy):
    def __init__(self):
        self.reps = np.ones(NUM_CLIENTS)
        self.global_w = None

    def initialize_parameters(self, client_manager): return None

    def configure_fit(self, rnd, params, cm): 
        clients = cm.sample(CLIENTS_PER_ROUND, CLIENTS_PER_ROUND)
        return clients, params, {}

    def aggregate_fit(self, rnd, results, failures):
        cids, deltas = zip(*[(int(r.client_id), r.parameters) for r in results])
        wlists = list(deltas)
        # Alg.2.1 Median reference
        stacked = [np.stack([w[layer] for w in wlists],0) for layer in range(len(wlists[0]))]
        U_med = [np.median(s,0) for s in stacked]
        # Alg.2.2 Distances
        ds = [np.sqrt(sum(np.linalg.norm(wl-um)**2 for wl,um in zip(w, U_med))) for w in wlists]
        D_max = max(max(ds), EPSILON)
        # Alg.2.3 Scores & reputations
        weights=[]
        for cid, d in zip(cids, ds):
            S = max(0, min(1,1 - d/D_max))
            self.reps[cid] = REP_DECAY*self.reps[cid] + (1-REP_DECAY)*S
            wi = self.reps[cid] if self.reps[cid]>=REP_THRESHOLD else 0
            weights.append(wi)
        w = np.array(weights)
        if w.sum()==0: w = np.ones_like(w)
        w /= w.sum()
        # Alg.2.4 Weighted aggregate
        new_g=[]
        for s in stacked:
            new_g.append(sum(w[i]*s[i] for i in range(len(w))))
        # Alg.3 Update
        if self.global_w is None:
            self.global_w = new_g
        else:
            self.global_w = [old+SERVER_LR*(ng-old) for old,ng in zip(self.global_w, new_g)]
        return self.global_w, {}

    def configure_evaluate(self, rnd, params, cm):
        clients = cm.sample(NUM_CLIENTS, NUM_CLIENTS)
        return clients, params, {}

    def aggregate_evaluate(self, rnd, results, failures):
        accs = [r.metrics["accuracy"] for r in results]
        return float(np.mean(accs)), {}

# ─── 7) LAUNCH & COMPARE ────────────────────────────────────────────────────────
if __name__ == "__main__":
    strategies = {
        "FedAvg": fl.server.strategy.FedAvg(),
        "Krum": fl.server.strategy.AggregateStrategy(
            aggregate_fit_fn=lambda r,rsl, f: (krum([x.parameters for x in rsl]), {}),
            evaluate_fn=None
        ),
        "Trimmed": fl.server.strategy.AggregateStrategy(
            aggregate_fit_fn=lambda r,rsl,f: (trimmed_mean([x.parameters for x in rsl]), {}),
            evaluate_fn=None
        ),
        "DRRA": FLIFRAStrategy(),
        "Hybrid": FLIFRAStrategy(),
    }

    histories = {}
    for name, strat in strategies.items():
        hist = fl.simulation.start_simulation(
            client_fn=lambda cid: fl.client.start_numpy_client("localhost:8080", client=None),
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=ROUNDS),
            strategy=strat,
        )
        histories[name] = hist.metrics_centralized["accuracy"]

    # Plot all
    plt.figure(figsize=(8,6))
    colors = {'FedAvg':'blue','Krum':'green','Trimmed':'red','DRRA':'purple','Hybrid':'orange'}
    for m, acc in histories.items():
        plt.plot(range(1, ROUNDS+1), acc, '--', marker='>', color=colors[m], label=m)
    plt.xlabel('Communication Rounds'); plt.ylabel('Test Accuracy')
    plt.ylim(0.7,1.0); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig('cicids_10percent_csf10.eps', format='eps')
    plt.show()
