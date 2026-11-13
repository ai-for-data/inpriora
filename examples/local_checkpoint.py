"""Run inference from a local converted checkpoint."""

import os

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from inpriora import InprioraRegressor

# Point this at a directory containing config.json and model.safetensors, or
# at a model.safetensors file.
model_path = os.environ["INPRIORA_MODEL_PATH"]

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
)

regressor = InprioraRegressor(model_path=model_path)
regressor.fit(X_train, y_train)
print(regressor.predict(X_test)[:5])