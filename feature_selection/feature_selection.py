import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
import warnings

warnings.filterwarnings("ignore")

# Set style
sns.set_theme(style="whitegrid")

# =============================================================================
# Create Sample Dataset (sentiment prediction based on age_group and other features)
# =============================================================================

np.random.seed(42)
n_samples = 1000

# Create features
data = pd.DataFrame(
    {
        "age_group": np.random.choice(
            ["18-25", "26-35", "36-45", "46-55", "56+"], n_samples
        ),
        "income_level": np.random.choice(["Low", "Medium", "High"], n_samples),
        "education": np.random.choice(
            ["High School", "Bachelor", "Master", "PhD"], n_samples
        ),
        "usage_frequency": np.random.randint(1, 100, n_samples),
        "session_duration": np.random.randint(5, 120, n_samples),
        "previous_purchases": np.random.randint(0, 50, n_samples),
        "support_tickets": np.random.randint(0, 10, n_samples),
    }
)

# Create target (sentiment) with some correlation to features
# Age_group has strong influence on sentiment
age_weight = {"18-25": 0.7, "26-35": 0.5, "36-45": 0.3, "46-55": 0.2, "56+": 0.1}
income_weight = {"Low": 0.2, "Medium": 0.5, "High": 0.8}

sentiment_score = (
    data["age_group"].map(age_weight) * 0.4  # age_group has 40% influence
    + data["income_level"].map(income_weight) * 0.3  # income has 30% influence
    + data["usage_frequency"] / 100 * 0.2  # usage has 20% influence
    + data["previous_purchases"] / 50 * 0.1  # purchases has 10% influence
    + np.random.random(n_samples) * 0.3  # Add noise
)

data["sentiment"] = (sentiment_score > 0.5).astype(
    int
)  # Binary: 0=Negative, 1=Positive

print("Dataset created!")
print(f"Shape: {data.shape}")
print(f"\nSentiment distribution:")
print(data["sentiment"].value_counts())
print(f"\nFirst few rows:")
print(data.head())

# Encode categorical variables
data_encoded = pd.get_dummies(data.drop("sentiment", axis=1), drop_first=False)
X = data_encoded
y = data["sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nFeatures after encoding: {X.columns.tolist()}")


# =============================================================================
# METHOD 1: Tree-Based Feature Importance (Random Forest)
# =============================================================================
print("\n" + "=" * 70)
print("METHOD 1: Random Forest Feature Importance")
print("=" * 70)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# Get feature importance
rf_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": rf_model.feature_importances_}
).sort_values("Importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(rf_importance.head(10))

# Calculate importance of age_group (sum of all age_group dummies)
age_features = [col for col in X.columns if "age_group" in col]
age_importance = rf_importance[rf_importance["Feature"].isin(age_features)][
    "Importance"
].sum()
print(f"\nTotal Age Group Importance: {age_importance:.4f}")

# Visualize
plt.figure(figsize=(12, 8))
top_features = rf_importance.head(15)
sns.barplot(data=top_features, y="Feature", x="Importance", palette="viridis")
plt.title(
    "Random Forest: Feature Importance (Gini-based)", fontsize=16, fontweight="bold"
)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)

# Highlight age_group features
for i, feature in enumerate(top_features["Feature"]):
    if "age_group" in feature:
        plt.gca().get_yticklabels()[i].set_color("red")
        plt.gca().get_yticklabels()[i].set_weight("bold")

plt.tight_layout()
plt.savefig("1_rf_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# METHOD 2: Gradient Boosting Feature Importance
# =============================================================================
print("\n" + "=" * 70)
print("METHOD 2: Gradient Boosting Feature Importance")
print("=" * 70)

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
gb_model.fit(X_train, y_train)

gb_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": gb_model.feature_importances_}
).sort_values("Importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(gb_importance.head(10))

age_importance_gb = gb_importance[gb_importance["Feature"].isin(age_features)][
    "Importance"
].sum()
print(f"\nTotal Age Group Importance: {age_importance_gb:.4f}")


# =============================================================================
# METHOD 3: Permutation Importance (Model-Agnostic)
# =============================================================================
print("\n" + "=" * 70)
print("METHOD 3: Permutation Importance (Most Reliable)")
print("=" * 70)

# Using Random Forest for permutation
perm_importance = permutation_importance(
    rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

perm_imp_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": perm_importance.importances_mean,
        "Std": perm_importance.importances_std,
    }
).sort_values("Importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(perm_imp_df.head(10))

age_importance_perm = perm_imp_df[perm_imp_df["Feature"].isin(age_features)][
    "Importance"
].sum()
print(f"\nTotal Age Group Importance: {age_importance_perm:.4f}")

# Visualize with error bars
plt.figure(figsize=(12, 8))
top_perm = perm_imp_df.head(15)
plt.barh(
    range(len(top_perm)),
    top_perm["Importance"],
    xerr=top_perm["Std"],
    color="skyblue",
    edgecolor="black",
)
plt.yticks(range(len(top_perm)), top_perm["Feature"])
plt.xlabel("Permutation Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Permutation Importance (with std deviation)", fontsize=16, fontweight="bold")
plt.gca().invert_yaxis()

# Highlight age_group features
for i, feature in enumerate(top_perm["Feature"]):
    if "age_group" in feature:
        plt.gca().get_yticklabels()[i].set_color("red")
        plt.gca().get_yticklabels()[i].set_weight("bold")

plt.tight_layout()
plt.savefig("2_permutation_importance.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# METHOD 4: Mutual Information (Statistical Dependency)
# =============================================================================
print("\n" + "=" * 70)
print("METHOD 4: Mutual Information Score")
print("=" * 70)

# Calculate mutual information
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

mi_df = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores}).sort_values(
    "MI_Score", ascending=False
)

print("\nTop 10 Most Important Features:")
print(mi_df.head(10))

age_importance_mi = mi_df[mi_df["Feature"].isin(age_features)]["MI_Score"].sum()
print(f"\nTotal Age Group MI Score: {age_importance_mi:.4f}")

# Visualize
plt.figure(figsize=(12, 8))
top_mi = mi_df.head(15)
sns.barplot(data=top_mi, y="Feature", x="MI_Score", palette="coolwarm")
plt.title("Mutual Information Scores", fontsize=16, fontweight="bold")
plt.xlabel("MI Score (Information Gain)", fontsize=12)
plt.ylabel("Feature", fontsize=12)

for i, feature in enumerate(top_mi["Feature"]):
    if "age_group" in feature:
        plt.gca().get_yticklabels()[i].set_color("red")
        plt.gca().get_yticklabels()[i].set_weight("bold")

plt.tight_layout()
plt.savefig("3_mutual_information.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# METHOD 5: Logistic Regression Coefficients
# =============================================================================
print("\n" + "=" * 70)
print("METHOD 5: Logistic Regression Coefficients")
print("=" * 70)

# Standardize features for fair comparison
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

lr_coef = pd.DataFrame(
    {
        "Feature": X.columns,
        "Coefficient": np.abs(lr_model.coef_[0]),  # Absolute value for importance
    }
).sort_values("Coefficient", ascending=False)

print("\nTop 10 Most Important Features:")
print(lr_coef.head(10))

age_importance_lr = lr_coef[lr_coef["Feature"].isin(age_features)]["Coefficient"].sum()
print(f"\nTotal Age Group Coefficient: {age_importance_lr:.4f}")


# =============================================================================
# COMPARISON: All Methods Together
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON: Age Group Importance Across All Methods")
print("=" * 70)

comparison = pd.DataFrame(
    {
        "Method": [
            "Random Forest (Gini)",
            "Gradient Boosting",
            "Permutation Importance",
            "Mutual Information",
            "Logistic Regression",
        ],
        "Age_Group_Importance": [
            age_importance,
            age_importance_gb,
            age_importance_perm,
            age_importance_mi,
            age_importance_lr,
        ],
    }
)

print("\n", comparison)

# Visualize comparison
plt.figure(figsize=(12, 6))
sns.barplot(
    data=comparison,
    x="Method",
    y="Age_Group_Importance",
    palette="Set2",
    edgecolor="black",
)
plt.title(
    "Age Group Importance: Comparison Across Methods", fontsize=16, fontweight="bold"
)
plt.xlabel("Method", fontsize=12)
plt.ylabel("Importance Score", fontsize=12)
plt.xticks(rotation=15, ha="right")

for i, v in enumerate(comparison["Age_Group_Importance"]):
    plt.text(
        i,
        v + 0.01,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("4_comparison_all_methods.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# BONUS: SHAP Values (Most Advanced Method)
# =============================================================================
print("\n" + "=" * 70)
print("BONUS: SHAP Values (Requires: pip install shap)")
print("=" * 70)

try:
    import shap

    # Create explainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values_to_use = shap_values[1]  # Use positive class
    else:
        shap_values_to_use = shap_values

    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_to_use, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("5_shap_importance.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSHAP values calculated successfully!")
    print("SHAP shows how much each feature contributes to predictions")

except ImportError:
    print("\nSHAP library not installed. Install with: pip install shap")
    print("SHAP provides the most accurate feature importance with explanations.")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)
print(
    """
WHICH METHOD TO USE?

1. **Permutation Importance** (RECOMMENDED)
   - Most reliable and model-agnostic
   - Shows real impact on predictions
   - Works with any model
   - Best for your use case!

2. **Random Forest / Gradient Boosting Importance**
   - Fast and built-in
   - Good for tree-based models
   - Can be biased toward high-cardinality features

3. **Mutual Information**
   - Captures non-linear relationships
   - Statistical measure of dependency
   - Good for feature selection

4. **Logistic Regression Coefficients**
   - Interpretable for linear relationships
   - Requires feature scaling
   - Best for linear models

5. **SHAP Values** (BEST but requires extra library)
   - Most comprehensive and accurate
   - Provides explanations for individual predictions
   - Industry standard for model interpretability

FOR YOUR CASE (age_group → sentiment):
Use Permutation Importance or SHAP for most reliable results!
"""
)

print("\nFeature Importance tells you:")
print("✓ Which features influence predictions most")
print("✓ Whether age_group matters for sentiment prediction")
print("✓ Which features you can remove without losing accuracy")
print("✓ How to improve your model by focusing on important features")
print("=" * 70)
