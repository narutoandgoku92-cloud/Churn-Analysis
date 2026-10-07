import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# Create the folder for saved charts
os.makedirs("visualizations", exist_ok=True)


# Load the dataset
df = pd.read_csv("archive (2)/telco.csv")


# Check the size of the dataset
print("\nDATASET SHAPE")
print(df.shape)


# See all the columns
print("\nALL COLUMNS")

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# Check the data types
print("\nDATA TYPES")
print(df.dtypes)


# Check for missing values
print("\nMISSING VALUES")
print(df.isnull().sum())


# Check for duplicate rows
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())


# See how many customers stayed or left
print("\nCHURN LABEL DISTRIBUTION")
print(df["Churn Label"].value_counts())


# Get the churn percentages
churn_percentages = (
    df["Churn Label"]
    .value_counts(normalize=True)
    .mul(100)
)

print("\nCHURN PERCENTAGES")
print(churn_percentages)


# Set the visual style
sns.set_theme(
    style="whitegrid",
    font_scale=1.05
)

plt.rcParams["figure.dpi"] = 120


# ==================================================
# EXPLORATORY DATA ANALYSIS
# ==================================================

# Customer churn distribution
plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x="Churn Label",
    hue="Churn Label",
    order=["No", "Yes"],
    palette=["#4C78A8", "#E45756"],
    legend=False
)

plt.title(
    "Customer Churn Distribution",
    fontsize=17,
    fontweight="bold",
    pad=15
)

plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.0f",
        padding=3
    )

plt.tight_layout()

plt.savefig(
    "visualizations/churn_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Churn rate by contract type
contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn Label"],
    normalize="index"
) * 100

print("\nCHURN RATE BY CONTRACT TYPE")
print(contract_churn.round(2))


contract_churn = contract_churn.reindex(
    columns=["No", "Yes"]
)

ax = contract_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 6),
    color=["#4C78A8", "#E45756"]
)

plt.title(
    "Churn Rate by Contract Type",
    fontsize=17,
    fontweight="bold",
    pad=15
)

plt.xlabel("Contract Type")
plt.ylabel("Percentage of Customers")

plt.xticks(rotation=0)

plt.legend(
    title="Churn Status",
    labels=["Stayed", "Churned"]
)

for container in ax.containers:
    labels = [
        f"{bar.get_height():.1f}%"
        if bar.get_height() >= 5
        else ""
        for bar in container
    ]

    ax.bar_label(
        container,
        labels=labels,
        label_type="center",
        fontsize=9
    )

plt.ylim(0, 105)

plt.tight_layout()

plt.savefig(
    "visualizations/churn_by_contract.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Group customers by how long they've been with the company
df["Tenure Group"] = pd.cut(
    df["Tenure in Months"],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=[
        "0–12 months",
        "13–24 months",
        "25–36 months",
        "37–48 months",
        "49–60 months",
        "61–72 months"
    ],
    include_lowest=True
)


# Check the churn rate for each tenure group
tenure_churn = pd.crosstab(
    df["Tenure Group"],
    df["Churn Label"],
    normalize="index"
) * 100

print("\nCHURN RATE BY TENURE GROUP")
print(tenure_churn.round(2))


tenure_churn = tenure_churn.reindex(
    columns=["No", "Yes"]
)

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=tenure_churn.reset_index(),
    x="Tenure Group",
    y="Yes",
    color="#E45756"
)

plt.title(
    "Churn Rate by Customer Tenure",
    fontsize=17,
    fontweight="bold",
    pad=15
)

plt.xlabel("Customer Tenure")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=0)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.ylim(
    0,
    max(tenure_churn["Yes"]) + 10
)

plt.tight_layout()

plt.savefig(
    "visualizations/churn_by_tenure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Compare monthly charges for customers who stayed and left
plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn Label",
    y="Monthly Charge",
    hue="Churn Label",
    order=["No", "Yes"],
    palette=["#4C78A8", "#E45756"],
    legend=False
)

plt.title(
    "Monthly Charges by Churn Status",
    fontsize=17,
    fontweight="bold",
    pad=15
)

plt.xlabel("Churn Status")
plt.ylabel("Monthly Charge")

plt.tight_layout()

plt.savefig(
    "visualizations/monthly_charges_by_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==================================================
# MACHINE LEARNING
# ==================================================

print("\n" + "=" * 60)
print("MACHINE LEARNING")
print("=" * 60)


# Columns we don't need for the model
drop_columns = [
    "Customer ID",
    "Churn Score",
    "Churn Category",
    "Churn Reason",
    "Customer Status",
    "Churn Label",
    "Tenure Group"
]


# Separate features and target
X = df.drop(columns=drop_columns)

y = df["Churn Label"].map({
    "No": 0,
    "Yes": 1
})


print("\nFEATURES")
print(X.shape)

print("\nTARGET")
print(y.value_counts())


# Find categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nCATEGORICAL FEATURES")
print(categorical_columns)

print("\nNUMERICAL FEATURES")
print(numerical_columns)


# Prepare numerical columns
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Prepare categorical columns
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# Combine both preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTRAINING DATA")
print(X_train.shape)

print("\nTEST DATA")
print(X_test.shape)


# Create the models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
}


results = []
trained_models = {}


# Train and evaluate each model
for model_name, model in models.items():

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


    # Train the model
    pipeline.fit(
        X_train,
        y_train
    )


    # Make predictions
    y_pred = pipeline.predict(
        X_test
    )


    # Get probabilities for ROC-AUC
    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]


    # Calculate metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")


    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Stayed",
                "Churned"
            ],
            zero_division=0
        )
    )


    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })


    trained_models[model_name] = pipeline


# ==================================================
# MODEL COMPARISON
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# Compare model performance
metrics_to_plot = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

comparison_data = results_df.set_index(
    "Model"
)[metrics_to_plot]


ax = comparison_data.plot(
    kind="bar",
    figsize=(11, 6),
    color=[
        "#4C78A8",
        "#F2CF5B",
        "#59A14F",
        "#E45756",
        "#B279A2"
    ]
)

plt.title(
    "Model Performance Comparison",
    fontsize=17,
    fontweight="bold",
    pad=15
)

plt.xlabel("Model")
plt.ylabel("Score")

plt.xticks(rotation=0)

plt.ylim(0, 1.05)

plt.legend(
    title="Metric",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "visualizations/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Find the best model
best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

best_result = results_df.iloc[0]


print("\nBEST MODEL")
print(best_model_name)

print(
    f"Accuracy:  {best_result['Accuracy']:.4f}"
)

print(
    f"Precision: {best_result['Precision']:.4f}"
)

print(
    f"Recall:    {best_result['Recall']:.4f}"
)

print(
    f"F1 Score:  {best_result['F1 Score']:.4f}"
)

print(
    f"ROC-AUC:   {best_result['ROC-AUC']:.4f}"
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

# Use the winning Logistic Regression model
logistic_pipeline = trained_models[
    "Logistic Regression"
]

y_pred = logistic_pipeline.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Stayed",
        "Churned"
    ],
    yticklabels=[
        "Stayed",
        "Churned"
    ],
    cbar=False,
    annot_kws={
        "fontsize": 14,
        "fontweight": "bold"
    }
)

plt.title(
    "Logistic Regression\nConfusion Matrix",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "visualizations/confusion_matrix_logistic_regression.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==================================================

rf_pipeline = trained_models[
    "Random Forest"
]

rf_model = rf_pipeline.named_steps[
    "model"
]


# Get the processed feature names
feature_names = (
    rf_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n" + "=" * 60)
print("TOP 20 FEATURES")
print("=" * 60)

print(
    feature_importance
    .head(20)
    .to_string(index=False)
)


# Show the top 10 features
top_features = (
    feature_importance
    .head(10)
    .sort_values(
        by="Importance",
        ascending=True
    )
)


plt.figure(figsize=(10, 7))

ax = sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature",
    color="#4C78A8"
)

plt.title(
    "Top Features Influencing Churn Prediction",
    fontsize=17,
    fontweight="bold",
    pad=15
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")


for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.3f",
        padding=3
    )


plt.tight_layout()

plt.savefig(
    "visualizations/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==================================================
# FINAL PROJECT SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print(
    f"Dataset size: {df.shape[0]:,} customers"
)

print(
    f"Number of features used: {X.shape[1]}"
)

print(
    f"Best model: {best_model_name}"
)

print(
    f"Accuracy: "
    f"{best_result['Accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{best_result['Precision'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{best_result['Recall'] * 100:.2f}%"
)

print(
    f"F1 Score: "
    f"{best_result['F1 Score'] * 100:.2f}%"
)

print(
    f"ROC-AUC: "
    f"{best_result['ROC-AUC'] * 100:.2f}%"
)


print(
    "\nTop factors associated with "
    "churn predictions:"
)

for _, row in feature_importance.head(5).iterrows():

    print(
        f"- {row['Feature']}: "
        f"{row['Importance']:.4f}"
    )


print(
    "\nCharts saved to the "
    "'visualizations' folder."
)

print(
    "\nProject completed successfully."
)