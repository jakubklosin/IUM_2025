import mlflow
import mlflow.pytorch

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

def main(epochs, lr, model_path, dataset_path):
    mlflow.set_experiment("airbnb_mlflow_experiment")

    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)

        # Wczytaj dane
        df = pd.read_csv(dataset_path)
        X = df.drop(columns=['price', 'id', 'name', 'host_id', 'host_name', 'last_review'])
        y = df["price"]

        # Podział danych
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
        X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # Definicja modelu
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Trening
        for epoch in range(epochs):
            model.train()
            preds = model(X_train)
            loss = criterion(preds, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Ewaluacja
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test)
            val_loss = criterion(val_preds, y_test).item()

        mlflow.log_metric("val_loss", val_loss)

        # Zapis modelu lokalnie i do MLflow
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"Training completed. Final val_loss: {val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--model_path", type=str, default="regression_model.pt")
    parser.add_argument("--dataset_path", type=str, default="airbnb_cleaned.csv")
    args = parser.parse_args()

    main(args.epochs, args.lr, args.model_path, args.dataset_path)
