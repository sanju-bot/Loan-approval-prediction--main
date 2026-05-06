import shap
import matplotlib.pyplot as plt
import seaborn as sns


def explain_model_with_shap(model, X_train, X_test, feature_names, output_dir='output/plots/'):
    """Generate SHAP explanations"""
    print("\n🔍 Generating SHAP Explanations...")

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, take the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # 1. Summary Plot
    print("  ✓ Creating SHAP Summary Plot...")
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}shap_summary_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: shap_summary_plot.png")

    # 2. Feature Importance Bar Plot
    print("  ✓ Creating Feature Importance Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}shap_feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: shap_feature_importance.png")

    # 3. Dependence Plot for top 3 features
    print("  ✓ Creating Dependence Plots...")
    mean_abs_shap = (shap_values ** 2).mean(axis=0)
    top_indices = sorted(range(len(mean_abs_shap)),
                         key=lambda i: mean_abs_shap[i],
                         reverse=True)[:3]

    for i, idx in enumerate(top_indices):
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(idx, shap_values, X_test,
                             feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(f'{output_dir}shap_dependence_{feature_names[idx]}.png',
                    dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: shap_dependence_{feature_names[idx]}.png")

    print("✓ SHAP Explanation Complete!")
    return explainer, shap_values


def explain_single_prediction(explainer, shap_values, X_test, feature_names, instance_idx=0):
    """Explain a single prediction"""
    print(f"\n🎯 Explaining Prediction for Instance {instance_idx}:")
    print("=" * 60)

    # Get SHAP values for this instance
    instance_shap = shap_values[instance_idx]

    # Get top contributing features
    top_indices = sorted(range(len(instance_shap)),
                         key=lambda i: abs(instance_shap[i]),
                         reverse=True)[:5]

    print(f"\nTop 5 Features Contributing to this Prediction:\n")
    for rank, idx in enumerate(top_indices, 1):
        feature = feature_names[idx]
        value = X_test.iloc[instance_idx, idx]
        shap_val = instance_shap[idx]
        direction = "↑" if shap_val > 0 else "↓"

        print(f"{rank}. {feature}: {value} ({direction} {shap_val:.4f})")

    print("=" * 60)