import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification

# Set style for better-looking plots
sns.set_theme(style="whitegrid", palette="muted")

# Create sample datasets
np.random.seed(42)

# Dataset 1: Model performance metrics
model_performance = pd.DataFrame(
    {
        "Model": [
            "Linear Reg",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "Neural Net",
        ],
        "Accuracy": [0.75, 0.82, 0.88, 0.91, 0.89],
        "Training_Time": [0.5, 2.1, 5.3, 3.8, 15.2],
    }
)

# Dataset 2: Feature importance
features = pd.DataFrame(
    {
        "Feature": ["Age", "Income", "Credit_Score", "Debt_Ratio", "Employment"],
        "Importance": [0.28, 0.35, 0.22, 0.10, 0.05],
    }
)

# Dataset 3: Training metrics over epochs
epochs = pd.DataFrame(
    {
        "Epoch": range(1, 51),
        "Training_Loss": np.exp(-np.linspace(0, 3, 50)) + np.random.normal(0, 0.02, 50),
        "Validation_Loss": np.exp(-np.linspace(0, 2.5, 50))
        + np.random.normal(0, 0.03, 50),
    }
)

# Dataset 4: Classification dataset for scatter plots
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42,
)
classification_data = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
classification_data["Class"] = y

# Dataset 5: Iris dataset for pairplot
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)


# =============================================================================
# 1. BAR CHART - Model Performance Comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=model_performance, x="Model", y="Accuracy", ax=ax, hue="Model", legend=False
)
ax.set_title("Model Accuracy Comparison", fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel("Model Type", fontsize=12)
ax.set_ylabel("Accuracy Score", fontsize=12)
ax.set_ylim(0.7, 0.95)

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)

plt.tight_layout()
plt.savefig("1_barplot_model_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# 2. HORIZONTAL BAR CHART - Feature Importance
# =============================================================================
# fig, ax = plt.subplots(figsize=(10, 6))
# features_sorted = features.sort_values("Importance")
# sns.barplot(
#     data=features_sorted,
#     y="Feature",
#     x="Importance",
#     ax=ax,
#     hue="Feature",
#     legend=False,
#     palette="viridis",
# )
# ax.set_title("Feature Importance Analysis", fontsize=16, fontweight="bold", pad=20)
# ax.set_xlabel("Importance Score", fontsize=12)
# ax.set_ylabel("Feature Name", fontsize=12)

# Add value labels
# for container in ax.containers:
#     ax.bar_label(container, fmt="%.2f", padding=5)

# plt.tight_layout()
# plt.savefig("2_horizontal_bar_feature_importance.png", dpi=300, bbox_inches="tight")
# plt.show()


# =============================================================================
# 3. LINE PLOT - Training Progress
# =============================================================================
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.lineplot(
#     data=epochs,
#     x="Epoch",
#     y="Training_Loss",
#     label="Training Loss",
#     linewidth=2.5,
#     marker="o",
#     markersize=4,
#     markevery=5,
#     ax=ax,
# )
# sns.lineplot(
#     data=epochs,
#     x="Epoch",
#     y="Validation_Loss",
#     label="Validation Loss",
#     linewidth=2.5,
#     marker="s",
#     markersize=4,
#     markevery=5,
#     ax=ax,
# )
# ax.set_title("Model Training Progress", fontsize=16, fontweight="bold", pad=20)
# ax.set_xlabel("Epoch", fontsize=12)
# ax.set_ylabel("Loss", fontsize=12)
# ax.legend(fontsize=11, loc="upper right")
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig("3_lineplot_training_progress.png", dpi=300, bbox_inches="tight")
# plt.show()


# =============================================================================
# 4. SCATTER PLOT - Classification Results
# =============================================================================
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.scatterplot(
#     data=classification_data,
#     x="Feature_1",
#     y="Feature_2",
#     hue="Class",
#     palette="Set1",
#     s=100,
#     alpha=0.7,
#     edgecolor="black",
#     linewidth=0.5,
#     ax=ax,
# )
# ax.set_title(
#     "Binary Classification - Feature Space", fontsize=16, fontweight="bold", pad=20
# )
# ax.set_xlabel("Feature 1", fontsize=12)
# ax.set_ylabel("Feature 2", fontsize=12)
# ax.legend(title="Class", fontsize=11, title_fontsize=12)

# plt.tight_layout()
# plt.savefig("4_scatterplot_classification.png", dpi=300, bbox_inches="tight")
# plt.show()


# =============================================================================
# 5. HISTOGRAM - Distribution Analysis
# =============================================================================
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.histplot(
#     data=iris_df,
#     x="sepal length (cm)",
#     hue="species",
#     kde=True,
#     bins=20,
#     alpha=0.6,
#     ax=ax,
# )
# ax.set_title(
#     "Distribution of Sepal Length by Species", fontsize=16, fontweight="bold", pad=20
# )
# ax.set_xlabel("Sepal Length (cm)", fontsize=12)
# ax.set_ylabel("Frequency", fontsize=12)
# ax.legend(title="Species", fontsize=10, title_fontsize=11)

# plt.tight_layout()
# plt.savefig("5_histogram_distribution.png", dpi=300, bbox_inches="tight")
# plt.show()


# =============================================================================
# 6. BOX PLOT - Outlier Detection
# =============================================================================
# fig, ax = plt.subplots(figsize=(12, 6))
# iris_melted = iris_df.melt(id_vars="species", var_name="Feature", value_name="Value")
# sns.boxplot(data=iris_melted, x="Feature", y="Value", hue="species", ax=ax)
# ax.set_title(
#     "Feature Distribution & Outliers by Species", fontsize=16, fontweight="bold", pad=20
# )
# ax.set_xlabel("Features", fontsize=12)
# ax.set_ylabel("Value (cm)", fontsize=12)
# ax.legend(title="Species", fontsize=10, title_fontsize=11, loc="upper right")
# plt.xticks(rotation=15)

# plt.tight_layout()
# plt.savefig("6_boxplot_outliers.png", dpi=300, bbox_inches="tight")
# plt.show()


# =============================================================================
# 7. HEATMAP - Correlation Matrix
# =============================================================================
# fig, ax = plt.subplots(figsize=(10, 8))
# correlation_matrix = iris_df.iloc[:, :-1].corr()
# sns.heatmap(
#     correlation_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="coolwarm",
#     center=0,
#     square=True,
#     linewidths=1,
#     cbar_kws={"shrink": 0.8},
#     ax=ax,
# )
# ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=20)

# plt.tight_layout()
# plt.savefig("7_heatmap_correlation.png", dpi=300, bbox_inches="tight")
# plt.show()


# =============================================================================
# 8. PAIRPLOT - Multi-dimensional Relationships
# =============================================================================
# pairplot = sns.pairplot(
#     iris_df,
#     hue="species",
#     diag_kind="kde",
#     plot_kws={"alpha": 0.6, "s": 50, "edgecolor": "k", "linewidth": 0.5},
#     height=2.5,
# )
# pairplot.fig.suptitle(
#     "Pairwise Feature Relationships", y=1.02, fontsize=16, fontweight="bold"
# )
# plt.savefig("8_pairplot_relationships.png", dpi=300, bbox_inches="tight")
# plt.show()


# =============================================================================
# 9. VIOLIN PLOT - Distribution with Density
# =============================================================================
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.violinplot(
#     data=iris_melted,
#     x="Feature",
#     y="Value",
#     hue="species",
#     split=False,
#     inner="box",
#     ax=ax,
# )
# ax.set_title(
#     "Feature Distribution Density by Species", fontsize=16, fontweight="bold", pad=20
# )
# ax.set_xlabel("Features", fontsize=12)
# ax.set_ylabel("Value (cm)", fontsize=12)
# ax.legend(title="Species", fontsize=10, title_fontsize=11, loc="upper right")
# plt.xticks(rotation=15)

# plt.tight_layout()
# plt.savefig("9_violinplot_density.png", dpi=300, bbox_inches="tight")
# plt.show()


# =============================================================================
# 10. CONFUSION MATRIX HEATMAP
# =============================================================================
# from sklearn.metrics import confusion_matrix

# Simulate predictions
# true_labels = np.random.randint(0, 3, 100)
# predicted_labels = true_labels.copy()
# Add some errors
# error_indices = np.random.choice(100, 15, replace=False)
# predicted_labels[error_indices] = np.random.randint(0, 3, 15)

# cm = confusion_matrix(true_labels, predicted_labels)
# cm_df = pd.DataFrame(
#     cm,
#     index=["Class 0", "Class 1", "Class 2"],
#     columns=["Class 0", "Class 1", "Class 2"],
# )

# fig, ax = plt.subplots(figsize=(8, 6))
# sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar_kws={"shrink": 0.8}, ax=ax)
# ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
# ax.set_xlabel("Predicted Label", fontsize=12)
# ax.set_ylabel("True Label", fontsize=12)

# plt.tight_layout()
# plt.savefig("10_confusion_matrix.png", dpi=300, bbox_inches="tight")
# plt.show()

# print("All visualizations created successfully!")
# print("\nBest Practices Applied:")
# print("1. Clear, descriptive titles and labels")
# print("2. Appropriate color palettes")
# print("3. Proper figure sizing")
# print("4. Value annotations where helpful")
# print("5. Legend positioning and clarity")
# print("6. High DPI for publication quality")
# print("7. Tight layout for clean presentation")
