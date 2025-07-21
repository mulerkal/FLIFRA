import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import flwr as fl

class FELACSClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]

        history = self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        parameters_prime = self.model.get_weights()
        results = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
        }
        return parameters_prime, len(self.x_train), results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        y_pred = np.argmax(self.model.predict(self.x_test), axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        report = classification_report(y_true, y_pred, output_dict=True)
        
        metrics = {"accuracy": report["accuracy"]}
        
        for i in range(8):  # 8 classes
            metrics[f"precision_class_{i}"] = report[str(i)]["precision"]
            metrics[f"recall_class_{i}"] = report[str(i)]["recall"]
            metrics[f"f1_score_class_{i}"] = report[str(i)]["f1-score"]

        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), metrics

def main() -> None:
    path = 'cic_ids2018_dataset.csv'
    data = pd.read_csv(path)
    X = data.drop(columns=["Label"]).to_numpy()
    y = pd.get_dummies(data['Label']).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    client = FELACSClient(model, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()
