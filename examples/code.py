#import sys, os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sys, os
print("Python path:", sys.path)
print("Current dir:", os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ===============================
# 1. Imports
# ===============================
from tracker_sdk.core import Tracker
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)

tracker = Tracker()
# Tracker function
_original_pipeline_fit = Pipeline.fit
_original_grid_fit = GridSearchCV.fit

def tracked_pipeline_fit(self, X,y,*args, **kwargs):
    # log dataset automatically
    
    tracker.log_dataset(X,y,name="dataset")
    print("Logging dataset:", name, "Shape:", X.shape)

    # Log Hyperparameters
    if hasattr(self, 'named_steps'):
        clf_params = self.named_steps['classifier'].get_params()
        tracker.start_experiment(self.named_steps['classifier'].__class__.__name__)
        tracker.log_hyperparameters(clf_params)
        print("Logging hyperparameters...")
    #call original fit
    result = _original_pipeline_fit(self, X,y,*args, **kwargs)
    return result

def tracked_grid_fit(self, X, y, *args, **kwargs):
    # Call original fit
    result = _original_grid_fit(self, X, y, *args, **kwargs)

    # Log best parameters and best score
    tracker.start_experiment(self.best_estimator_.named_steps['classifier'].__class__.__name__)
    tracker.log_hyperparameters(self.best_params_)
    from sklearn.metrics import accuracy_score
    y_pred = self.predict(X)
    tracker.log_metric("grid_train_accuracy", accuracy_score(y, y_pred))
    print("Logging metric...")
    return result

# Apply monkey patch
Pipeline.fit = tracked_pipeline_fit
GridSearchCV.fit = tracked_grid_fit
# ===============================
# 2. Load Data
# ===============================
column_names = ["id", "age","income","home_ownership_status","employment_length_in_years",
                "intent_of_loan","loan_amount","interest_rate","loan_approval_status", 
                "percent_income","default_status","length_of_credit_history"]

data = pd.read_csv('./storage/datasets/credit_risk.csv', header=None, names=column_names, skiprows=[0])
data = data.dropna()  # Drop rows with missing values (or impute if needed)

# ===============================
# 3. Define features and target
# ===============================
# Drop columns that are not features
X = data.drop(columns=["id", "loan_approval_status"])
y = data["loan_approval_status"]

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ===============================
# 4. Preprocessing
# ===============================
# Numeric: scale
numeric_transformer = StandardScaler()

# Categorical: OneHotEncoding (drop first to avoid multicollinearity)
categorical_transformer = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# ===============================
# 5. Feature Selection using Mutual Information
# ===============================
# We can select top k features based on MI
mi_selector = SelectKBest(mutual_info_classif, k='all')  # For now, keep all; you can limit later

# ===============================
# 6. Define Models
# ===============================
models = {
    "LogisticRegression": LogisticRegression(max_iter=500, solver='lbfgs', class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=10, subsample=0.6, random_state=42)
}

# ===============================
# 7. Split Data (Stratified)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 8. Create Pipelines & Train
# ===============================
pipelines = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', mi_selector),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    pipelines[name] = pipeline

# ===============================
# 9. Evaluate Models
# ===============================
for name, pipeline in pipelines.items():
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    tracker.start_experiment(pipelines[name])
    print(f"[INFO] Training pipeline: {name}")
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
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    tracker.log_metric(f"{name}_accuracy", acc)
    tracker.log_metric(f"{name}_precision", prec)
    tracker.log_metric(f"{name}_recall", rec)
    tracker.log_metric(f"{name}_f1_score", f1)
    tracker.log_metric(f"{name}_roc_auc", roc)
    
    # Optional: plot confusion matrix
    #ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)

# ===============================
# 10. Hyperparameter Tuning Example (RandomForest)
# ===============================
#param_grid_rf = {
#    'classifier__n_estimators': [100, 200, 300],
#    'classifier__max_depth': [None, 5, 10],
#    'classifier__min_samples_split': [2, 5],
#    'classifier__min_samples_leaf': [1, 2]
#}

#grid_rf = GridSearchCV(
#    pipelines["RandomForest"],
#    param_grid_rf,
#    scoring='recall',  # optimize for recall (important for default detection)
#    cv=2,
#    n_jobs=-1
#)

#grid_rf.fit(X_train, y_train)
#best_rf = grid_rf.best_estimator_
#print("Best RandomForest Parameters:", grid_rf.best_params_)
