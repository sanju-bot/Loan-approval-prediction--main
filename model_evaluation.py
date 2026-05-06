from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns


def evaluate_models(models_dict, X_test, y_test):
    """Evaluate all models and return metrics"""
    print("\n📊 Model Evaluation Results:")
    print("=" * 70)

    results = {}
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }

        print(f"\n{name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

    print("=" * 70)
    return results


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """Plot confusion matrix"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'output/plots/confusion_matrix_{model_name}.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: confusion_matrix_{model_name}.png")


def get_best_model(results):
    """Find best model by accuracy"""
    best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
    return best_model_name