import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate

def random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report

def decision_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report

def svm(X_train, X_test, y_train, y_test):
    clf = SVC(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report

def logistic_regression(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report

def gradient_boosting(X_train, X_test, y_train, y_test):
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report

def print_combined_report(reports):
    headers = ["Metric"] + list(reports.keys())
    table = []

    for key in reports[list(reports.keys())[0]].keys():
        if isinstance(reports[list(reports.keys())[0]][key], dict):
            for sub_key in reports[list(reports.keys())[0]][key].keys():
                row = [f"{key} {sub_key}"]
                for model in reports.keys():
                    row.append(reports[model][key][sub_key])
                table.append(row)
        else:
            row = [key]
            for model in reports.keys():
                row.append(reports[model][key])
            table.append(row)

    print("Classification Report Comparison:\n")
    print(tabulate(table, headers, floatfmt=".2f"))

# Load the dataset
data_path = '/workspaces/advanced-ml-course/disease-symptoms/data/dt1.csv'
data = pd.read_csv(data_path)

# Preprocess the data (convert categorical variables to numerical)
data = pd.get_dummies(data, drop_first=True)

# Split the data into features and target variable
X = data.drop('Outcome Variable_Positive', axis=1)
y = data['Outcome Variable_Positive']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run all models
models = {
    "Random Forest": random_forest,
    "Decision Tree": decision_tree,
    "SVM": svm,
    "Logistic Regression": logistic_regression,
    "Gradient Boosting": gradient_boosting
}

reports = {}
for model_name, model_func in models.items():
    accuracy, report = model_func(X_train, X_test, y_train, y_test)
    print(f"{model_name} Accuracy:", accuracy)
    reports[model_name] = report

# Print combined classification report
print_combined_report(reports)