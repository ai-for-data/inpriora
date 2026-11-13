"""Regression with InprioraRegressor."""

import os

from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from inpriora import InprioraRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
)

regressor = InprioraRegressor(
    n_estimators=8,
    temperature=1.0,
    random_seed=0,
    model_path=os.environ.get("INPRIORA_MODEL_PATH"),
)
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)
summaries = regressor.predict(
    X_test,
    output_type=["mean", "median", "variance"],
)

print("rmse:", mean_squared_error(y_test, predictions) ** 0.5)
print("median:", summaries["median"][:5])
print("variance:", summaries["variance"][:5])

# Quantiles are read from the predicted bucket distribution.
quantiles = regressor.predict(
    X_test,
    output_type="quantiles",
    alphas=[0.1, 0.5, 0.9],
)
print("quantiles:", quantiles[:3])