import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

# 1. Charger le dataset
data = pd.read_csv("/home/fixot-brendan/Desktop/decision_tree_ensemble_model/datasets/processed/cleaned_data.csv")  # ajuste le chemin si besoin

# 2. Séparer X et y
X = data.drop(columns=["performance"])  # <-- ✅ le bon nom
y = data["performance"]

# 3. Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Définir le modèle LightGBM
model = lgb.LGBMRegressor(
    n_estimators=75,
    max_depth=15,
    learning_rate=0.07,
    min_child_samples=3,
    reg_lambda=1.0,
    reg_alpha=0.05,
    random_state=42
)

# 5. Entraîner le modèle (avec timer)
start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()

train_time = end_train - start_train
print(f"⏱️ Training time: {train_time:.4f} seconds")

# 6. Prédire (avec timer)
start_pred = time.time()
y_pred = model.predict(X_test)
end_pred = time.time()

predict_time = end_pred - start_pred
print(f"⏱️ Prediction time: {predict_time:.4f} seconds")

# 7. Évaluer
mse = mean_squared_error(y_test, y_pred)
print(f"✅ MSE LightGBM: {mse:.6f}")

# 8. Afficher l'importance des features
lgb.plot_importance(model, max_num_features=10)
plt.title("LightGBM Feature Importance")
plt.show()
