import pandas as pd
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import json

df = pd.read_csv("data/features.csv")
features = ["c90","c252","c30","name","liq",
            "reg_mis","sty_mis","thm_mis","mod_mis","bond_mis"]

X, y = df[features].fillna(0), df["y"]
model = LogisticRegression(max_iter=500, solver="lbfgs").fit(X, y)

coef = dict(zip(features, model.coef_[0]))
inter = model.intercept_[0]

Path("data/weights.json").write_text(json.dumps({"intercept":inter, **coef}, indent=2))
print("✅  saved learnt weights → data/weights.json")
