# ===============================
# 0. Setup paths
# ===============================
import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Add project root to path so tracker_sdk can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ===============================
# 1. Import Tracker
# ===============================
from tracker_sdk.core import Tracker

tracker = Tracker(api_url="http://localhost:8000/model-save/")

# ===============================
# 2. Monkey patch for runtime logging
# ===============================
_original_pipeline_fit = Pipeline.fit
_original_grid_fit = GridSearchCV.fit

def tracked_pipeline_fit(self, X, y, *args, **kwargs):
    # Log dataset
    tracker.log_dataset(X, y, name="dataset_runtime")
    print("[Tracker] Dataset logged:", X.shape)

    # Log hyperparameters if classifier exists
    if hasattr(self, 'named_steps') and 'classifier' in self.named_steps:
        clf_params = self.named_steps['classifier'].get_params()
        tracker.start_experiment(self.named_steps['classifier'].__class__.__name__)
        tracker.log_hyperparameters(clf_params)
        print("[Tracker] Hyperparameters logged:", clf_params)

    # Call original fit
    return _original_pipeline_fit(self, X, y, *args, **kwargs)

def tracked_grid_fit(self, X, y, *args, **kwargs):
    # Fit normally
    result = _original_grid_fit(self, X, y, *args, **kwargs)

    # Log GridSearch best parameters and train accuracy
    tracker.start_experiment(self.best_estimator_.named_steps['classifier'].__class__.__name__)
    tracker.log_hyperparameters(self.best_params_)
    y_pred = self.predict(X)
    tracker.log_metric("grid_train_accuracy", accuracy_score(y, y_pred))
    print("[Tracker] GridSearch logged:", self.best_params_)
    return result

Pipeline.fit = tracked_pipeline_fit
GridSearchCV.fit = tracked_grid_fit

# ===============================
# 3. Load Data
# ===============================
column_names = ["id", "age","income","home_ownership_status","employment_length_in_years",
                "intent_of_loan","loan_amount","interest_rate","loan_approval_status", 
                "percent_income","default_status","length_of_credit_history"]

data = pd.read_csv('./storage/datasets/credit_risk.csv', header=None, names=column_names, skiprows=[0])
data = data.dropna()

X = data.drop(columns=["id", "loan_approval_status"])
y = data["loan_approval_status"]

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ===============================
# 4. Preprocessing
# ===============================
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

mi_selector = SelectKBest(mutual_info_classif, k='all')

# ===============================
# 5. Define Models
# ===============================
models = {
    "LogisticRegression": LogisticRegression(max_iter=500, solver='lbfgs', class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}

# ===============================
# 6. Train & Track Pipelines
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipelines = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', mi_selector),
        ('classifier', model)
    ])
    print(f"[INFO] Training pipeline: {name}")
    pipeline.fit(X_train, y_train)  # <-- Tracker logs dataset & hyperparameters automatically
    pipelines[name] = pipeline

# ===============================
# 7. Evaluate & Track Metrics
# ===============================
for name, pipeline in pipelines.items():
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"--- {name} ---")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)
    print("ROC-AUC:", roc)

    # Log metrics automatically
    tracker.log_metric("accuracy", acc)
    tracker.log_metric("precision", prec)
    tracker.log_metric("recall", rec)
    tracker.log_metric("f1", f1)
    tracker.log_metric("roc_auc", roc)

    # Confusion Matrix optional
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    #ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)

# ===============================
# 8. Optional: GridSearch with Tracker
# ===============================
#param_grid_rf = {
#    'classifier__n_estimators': [100, 200],
#    'classifier__max_depth': [None, 5]
#}

#grid_rf = GridSearchCV(pipelines["RandomForest"], param_grid_rf, scoring='recall', cv=2, n_jobs=-1)
#grid_rf.fit(X_train, y_train)  # <-- Tracker logs best params & train accuracy
#best_rf = grid_rf.best_estimator_
#rint("[INFO] GridSearch best params:", grid_rf.best_params_)
