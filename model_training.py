from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd


def prepare_features_target(df, target_column='Loan_Status'):
    """Separate features and target"""
    print(f"\n📊 Preparing Features & Target")
    print(f"Available columns: {df.columns.tolist()}")

    # ✅ Check if target exists
    if target_column not in df.columns:
        raise ValueError(f"❌ '{target_column}' not found in dataframe! Columns: {df.columns.tolist()}")

    print(f"✓ Found target column: {target_column}")

    # ✅ DROP loan_id AND target column
    X = df.drop([target_column, 'loan_id'], axis=1, errors='ignore')
    y = df[target_column]

    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Feature columns: {X.columns.tolist()}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Target distribution:\n{y.value_counts()}")

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split into train and test sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test):
    """Normalize features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_models(X_train, y_train):
    """Train multiple models"""
    print("\n🤖 Training Models...")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"  ✓ {name} trained successfully")
        except Exception as e:
            print(f"  ❌ {name} failed: {str(e)}")

    return trained_models


def tune_decision_tree(X_train, y_train):
    """Hyperparameter tuning for Decision Tree"""
    print("\n🔧 Tuning Decision Tree...")

    params = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print(f"✓ Best Parameters: {grid.best_params_}")
    print(f"✓ Best CV Score: {grid.best_score_:.4f}")

    return grid.best_estimator_


def tune_random_forest(X_train, y_train):
    """Hyperparameter tuning for Random Forest"""
    print("\n🔧 Tuning Random Forest...")

    params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print(f"✓ Best Parameters: {grid.best_params_}")
    print(f"✓ Best CV Score: {grid.best_score_:.4f}")

    return grid.best_estimator_


def perform_cross_validation(model, X_train, y_train, cv=5):
    """Perform cross-validation"""
    print(f"\n📊 Performing {cv}-Fold Cross-Validation...")

    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    print(f"✓ CV Scores: {scores}")
    print(f"✓ Mean CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return scores