import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score

df = pd.read_csv("airbnb_cleaned.csv")
X = df.drop(columns=['price', 'id', 'name', 'host_id', 'host_name', 'last_review'])
y = df['price']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(X_test.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)


model.load_state_dict(torch.load("regression_model.pt"))
model.eval()

# Wczytanie min i max ceny
with open("price_minmax.txt", "r") as f:
    lines = f.readlines()
    price_min = float(lines[0].strip())
    price_max = float(lines[1].strip())

# Predykcja
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze().numpy()

# Odwrotna normalizacja
predictions_original = predictions * (price_max - price_min) + price_min

# Zapis wyników
output_df = pd.DataFrame({"PredictedPrice": predictions_original})
output_df.to_csv("airbnb_predictions.csv", index=False)
print("Predykcje zapisane do pliku airbnb_predictions.csv")

y_test = y_test.reset_index(drop=True)
y_test_original = y_test * (price_max - price_min) + price_min

# Obliczenie metryk
rmse = root_mean_squared_error(y_test_original, predictions_original)
r2 = r2_score(y_test_original, predictions_original)

# Zapis metryk do pliku
with open("metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")
