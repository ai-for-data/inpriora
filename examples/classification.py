"""Binary classification with InprioraClassifier."""

import os

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
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

classifier = InprioraClassifier(
    n_estimators=8,
    temperature=1.0,
    random_seed=0,
    model_path=os.environ.get("INPRIORA_MODEL_PATH"),
)
classifier.fit(X_train, y_train)

probabilities = classifier.predict_proba(X_test)
predictions = classifier.predict(X_test)

print("classes:", classifier.classes_)
print("accuracy:", accuracy_score(y_test, predictions))
print("roc auc:", roc_auc_score(y_test, probabilities[:, 1]))