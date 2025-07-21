# Client 

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import flwr as fl
import keras
from keras import layers, models

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_PATH     = "cicids_2018.csv"
LABEL_COLUMN  = "attack_label"
NUM_CLASSES   = 8          # number of label categories
IF_TREES      = 100        # number of trees
CONTAM_IF     = 0.10       # contamination η
EPOCHS        = 100
BATCH_SIZE    = 64

# ─── DATA LOADING & PREPROCESSING ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.dropna().drop_duplicates(inplace=True)
le = LabelEncoder()
df[LABEL_COLUMN] = le.fit_transform(df[LABEL_COLUMN])
y = keras.utils.to_categorical(df[LABEL_COLUMN], num_classes=NUM_CLASSES)
X = df.drop(columns=[LABEL_COLUMN]).values
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=df[LABEL_COLUMN], random_state=42
)

# ─── MODEL FACTORY ───────────────────────────────────────────────────────────────
def create_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ─── FLOWER CLIENT ───────────────────────────────────────────────────────────────
class FLIFRAClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = int(cid)

        # split data 
        idxs = np.array_split(np.arange(len(X_train)), 10)
        self.X, self.y = X_train[idxs[self.cid]], y_train[idxs[self.cid]]

        self.model = create_model(X_train.shape[1], NUM_CLASSES)

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        #set global model
        self.model.set_weights(parameters)

        # iForest anomaly detection & filtering
        iso = IsolationForest(
            n_estimators=IF_TREES,
            contamination=CONTAM_IF,
            random_state=42,
        )
        iso.fit(self.X)
        mask = iso.predict(self.X) == 1
        X_filt, y_filt = self.X[mask], self.y[mask]

        # train on filtered data
        self.model.fit(
            X_filt,
            y_filt,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        #return update Δ = M_i − M
        new_w = self.model.get_weights()
        delta = [nw - w for nw, w in zip(new_w, parameters)]
        return delta, len(X_filt), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        return float(loss), len(X_test), {"accuracy": float(acc)}


if __name__ == "__main__":
    # Each client will connect to localhost:8080 by default
    fl.client.start_numpy_client(server_address="localhost:8080", client=FLIFRAClient)


