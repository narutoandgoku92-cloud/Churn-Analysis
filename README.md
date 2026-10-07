# Customer Churn Prediction

A machine learning project that analyzes customer behavior and predicts whether a customer is likely to churn.

The project uses a Telco Customer Churn dataset containing information about customer demographics, services, contracts, charges, tenure, satisfaction, and other customer-related attributes.

The workflow covers data inspection, exploratory data analysis, preprocessing, model training, model evaluation, and feature-importance analysis.

---

## Project Overview

Customer churn is an important problem for subscription-based businesses.

When customers leave a service, businesses lose recurring revenue and may need to spend additional resources acquiring new customers.

The goal of this project is to:

* Explore customer churn patterns
* Identify factors associated with churn
* Prepare customer data for machine learning
* Train multiple classification models
* Compare model performance
* Identify the strongest-performing model
* Analyze which features influenced the predictions most

---

## Dataset

The dataset used for this project is the **Telco Customer Churn** dataset from Kaggle.

**Dataset source:**

https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3

The dataset contains:

* **7,043 customers**
* **50 columns**

The target variable is:

```text
Churn Label
```

It contains two possible values:

* `No` — customer stayed
* `Yes` — customer churned

### Dataset information

The dataset includes information about:

* Customer demographics
* Customer location
* Dependents
* Referrals
* Tenure
* Contract type
* Internet services
* Phone services
* Payment methods
* Monthly charges
* Total charges
* Satisfaction score
* Customer lifetime value
* Churn information

---

## Project Structure

```text
customer-churn-prediction/
│
├── archive (2)/
│   └── telco.csv
│
├── customer_churn.py
│
├── visualizations/
│   ├── churn_distribution.png
│   ├── churn_by_contract.png
│   ├── churn_by_tenure.png
│   ├── monthly_charges_by_churn.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
└── README.md
```

> The exact folder structure may differ depending on how the project is organized locally.

---

# Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

### Machine Learning

* Logistic Regression
* Decision Tree
* Random Forest

### Development Environment

* Visual Studio Code
* Python virtual environment

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

Move into the project directory:

```bash
cd customer-churn-prediction
```

---

## 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install the required libraries

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

# Running the Project

Make sure the dataset is available at the path expected by the Python script.

Then run:

```bash
python customer_churn.py
```

The program will:

1. Load the dataset
2. Inspect the data
3. Check missing values
4. Check duplicate rows
5. Analyze churn distribution
6. Generate exploratory visualizations
7. Prepare the data for machine learning
8. Train three classification models
9. Evaluate the models
10. Compare their performance
11. Generate confusion matrices
12. Analyze Random Forest feature importance
13. Display a final project summary

---

# Data Inspection

The first stage of the project involved understanding the dataset before building any machine learning models.

The dataset contains:

```text
7,043 rows
50 columns
```

### Missing values

Several columns contained missing values, including:

* `Offer`
* `Internet Type`
* `Churn Category`
* `Churn Reason`

These missing values were handled during the machine learning preprocessing stage.

### Duplicate rows

The dataset contained:

```text
0 duplicate rows
```

---

# Exploratory Data Analysis

Several visualizations were created to understand customer churn patterns.

## Customer Churn Distribution

The dataset contains more customers who stayed than customers who churned.

Approximately:

* **73.46% stayed**
* **26.54% churned**

This shows that the target variable is somewhat imbalanced, which is why metrics such as recall, F1-score, and ROC-AUC were considered alongside accuracy.

---

## Churn Rate by Contract Type

Customers with different contract types showed different churn rates.

Contract type was one of the important variables when examining customer retention patterns.

The analysis compares:

* Month-to-month contracts
* One-year contracts
* Two-year contracts

---

## Churn Rate by Customer Tenure

Customers were grouped according to how long they had been with the company:

* 0–12 months
* 13–24 months
* 25–36 months
* 37–48 months
* 49–60 months
* 61–72 months

This helps show how churn behavior changes as customer tenure increases.

---

## Monthly Charges by Churn Status

Monthly charges were compared between customers who stayed and customers who churned.

A boxplot was used to examine the distribution and spread of monthly charges for the two groups.

---

# Data Preprocessing

Before training the models, the dataset was prepared using a Scikit-learn preprocessing pipeline.

## Columns excluded from the model

The following columns were removed:

```text
Customer ID
Churn Score
Churn Category
Churn Reason
Customer Status
Churn Label
Tenure Group
```

### Why?

`Customer ID` is an identifier rather than a useful predictive feature.

`Churn Score`, `Churn Category`, and `Churn Reason` contain information directly related to churn and could cause **data leakage**.

`Customer Status` was also excluded because it contains information closely related to the target.

`Tenure Group` was created for exploratory analysis and was excluded so the model could use the original `Tenure in Months` variable instead.

---

# Handling Numerical Features

Numerical features were processed using:

```python
SimpleImputer(strategy="median")
```

to handle missing numerical values.

The numerical variables were then standardized using:

```python
StandardScaler()
```

---

# Handling Categorical Features

Categorical features were processed using:

```python
SimpleImputer(strategy="most_frequent")
```

Missing categorical values were replaced with the most common value.

Categorical variables were then converted into numerical representations using:

```python
OneHotEncoder(handle_unknown="ignore")
```

---

# Train-Test Split

The dataset was divided into:

* **80% training data**
* **20% testing data**

The split used stratification to maintain a similar churn distribution in both sets.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The test set contained:

```text
1,409 customers
```

---

# Machine Learning Models

Three classification algorithms were trained and compared.

## Logistic Regression

Logistic Regression was used as a baseline classification model.

It performed extremely well on this dataset and ultimately achieved the strongest overall results.

---

## Decision Tree

A Decision Tree was trained with a maximum depth of 6.

Decision Trees can capture non-linear relationships and are relatively easy to interpret.

---

## Random Forest

A Random Forest consisting of 200 trees was trained.

Random Forest combines multiple decision trees to make predictions.

It was also used to examine feature importance.

---

# Model Evaluation

The models were evaluated using:

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many customers predicted as churners actually churned.

### Recall

Measures how many of the customers who actually churned were successfully identified.

### F1 Score

Provides a balance between precision and recall.

### ROC-AUC

Measures how well the model separates customers who churn from customers who stay across different classification thresholds.

---

# Model Results

The final results were:

| Model                   |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression** | **96.17%** |     95.45% | **89.84%** | **92.56%** | **99.19%** |
| Decision Tree           |     95.10% | **98.72%** |     82.62% |     89.96% |     98.62% |
| Random Forest           |     89.64% |     95.97% |     63.64% |     76.53% |     96.06% |

---

# Best Performing Model

Based on the evaluation results, **Logistic Regression** was the strongest-performing model.

Its performance was:

```text
Accuracy:   96.17%
Precision:  95.45%
Recall:     89.84%
F1 Score:   92.56%
ROC-AUC:    99.19%
```

The model successfully identified a large proportion of customers who actually churned while maintaining high precision.

Because customer churn is the main target of the project, recall is particularly useful because failing to identify a customer who is likely to churn could mean losing an opportunity for customer retention.

---

# Confusion Matrix

Confusion matrices were generated for all three models.

They show:

* True Negatives — customers correctly predicted to stay
* False Positives — customers predicted to churn who stayed
* False Negatives — customers predicted to stay who churned
* True Positives — customers correctly predicted to churn

The confusion matrices provide a more detailed view of where each model succeeds and makes mistakes.

---

# Feature Importance

Random Forest feature importance was used to examine which features had the greatest influence on its predictions.

The strongest features included:

| Feature                     | Importance |
| --------------------------- | ---------: |
| Satisfaction Score          |     0.2421 |
| Contract — Month-to-Month   |     0.0585 |
| Tenure in Months            |     0.0482 |
| Contract — Two Year         |     0.0406 |
| Number of Referrals         |     0.0394 |
| Total Revenue               |     0.0348 |
| Monthly Charge              |     0.0306 |
| Total Charges               |     0.0289 |
| Total Long Distance Charges |     0.0244 |
| Average Monthly GB Download |     0.0240 |

### Important interpretation

Feature importance indicates which variables were influential in the model's predictions.

It **does not prove that a feature causes churn**.

For example, Satisfaction Score being the strongest feature does not mean that satisfaction directly causes 24.21% of churn.

A more accurate interpretation is:

> Satisfaction Score was the most influential feature in the Random Forest model's predictions.

---

# Key Findings

Several patterns stood out during the analysis.

### Customer satisfaction

Satisfaction Score was the strongest feature in the Random Forest's predictions.

This suggests that customer experience is an important signal when identifying potential churn.

### Contract type

Contract type was one of the strongest categorical features.

Customers on different contract types showed noticeably different churn behavior.

### Customer tenure

Tenure was another important feature.

The churn rate varied across different customer-tenure groups, showing that customer retention behavior changes over time.

### Charges and revenue

Monthly Charge, Total Charges, and Total Revenue were also among the influential numerical features.

These variables provide information about the customer's financial relationship with the service.

---

# Limitations

This project has several limitations.

### Dataset limitations

The model was trained and evaluated on a single publicly available dataset.

Performance on another company's customer base may be different.

### Feature importance is not causation

The feature-importance analysis identifies influential predictive variables but does not establish causal relationships.

### Model generalization

The reported results come from a train-test split of the available dataset.

Further validation using cross-validation and external datasets would provide stronger evidence about generalization.

### Geographic features

The dataset contains geographic information such as:

* State
* City
* ZIP Code
* Latitude
* Longitude

These variables may introduce high-cardinality features and may not always be appropriate for a real-world churn model without additional consideration.

---

# Future Improvements

Possible improvements include:

* Cross-validation
* Hyperparameter tuning
* Feature selection
* Testing additional algorithms
* Handling class imbalance with techniques such as class weights
* Threshold optimization
* ROC and Precision-Recall curves
* Model explainability using SHAP
* Deployment as an API
* Building a simple customer churn prediction dashboard
* Testing the model on new customer data

---

# What I Learned

Through this project, I practiced:

* Working with a real-world dataset
* Data inspection and validation
* Handling missing values
* Exploratory data analysis
* Data visualization
* Feature preprocessing
* One-hot encoding
* Feature scaling
* Train-test splitting
* Classification algorithms
* Model evaluation
* Confusion matrices
* Feature importance
* Comparing multiple machine learning models
* Avoiding data leakage
* Interpreting machine learning results

---

# Conclusion

This project demonstrates a complete machine learning workflow for customer churn prediction.

After comparing Logistic Regression, Decision Tree, and Random Forest models, Logistic Regression achieved the strongest overall performance with:

```text
96.17% Accuracy
95.45% Precision
89.84% Recall
92.56% F1 Score
99.19% ROC-AUC
```

The analysis also highlighted satisfaction, contract type, tenure, referrals, and financial variables as important features in the model's predictions.

The project provided practical experience in taking a real-world dataset from initial exploration through preprocessing, model training, evaluation, and interpretation.

---

## Author

**Tomiwa Sijuwola**

Computer Science Student
Interested in Python, Machine Learning, Backend Development, and Software Development.

---

## Dataset Credit

Dataset: Telco Customer Churn

Source: Kaggle

https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3
