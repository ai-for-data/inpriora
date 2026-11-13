"""Reuse the Context cache across predict calls."""

import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from inpriora import InprioraClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
    stratify=y,
)

# cache=True keeps the key/value Context cache for later calls. Use "repr"
# for the smaller representation cache, or False to rebuild each time.
classifier = InprioraClassifier(
    n_estimators=8,
    cache=True,
    model_path=os.environ.get("INPRIORA_MODEL_PATH"),
)
classifier.fit(X_train, y_train)

first = classifier.predict_proba(X_test)
second = classifier.predict_proba(X_test)

print("identical:", (first == second).all())