import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# Variable to control whether to show confusion matrix
show_confusion_matrix = True

def random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    if show_confusion_matrix:
        print("Confusion Matrix for Random Forest:")
        print(confusion_matrix(y_test, y_pred))
    return accuracy, report

def decision_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    if show_confusion_matrix:
        print("Confusion Matrix for Decision Tree:")
        print(confusion_matrix(y_test, y_pred))
    return accuracy, report

def svm(X_train, X_test, y_train, y_test):
    clf = SVC(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    if show_confusion_matrix:
        print("Confusion Matrix for SVM:")
        print(confusion_matrix(y_test, y_pred))
    return accuracy, report

def logistic_regression(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    if show_confusion_matrix:
        print("Confusion Matrix for Logistic Regression:")
        print(confusion_matrix(y_test, y_pred))
    return accuracy, report

def gradient_boosting(X_train, X_test, y_train, y_test):
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    if show_confusion_matrix:
        print("Confusion Matrix for Gradient Boosting:")
        print(confusion_matrix(y_test, y_pred))
    return accuracy, report

def print_combined_report(reports, dataset_name):
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

    print(f"Classification Report Comparison for {dataset_name}:\n")
    print(tabulate(table, headers, floatfmt=".2f"))

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def load_and_preprocess_data(data_path, target_column):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Ensure the target column exists before processing
    if target_column not in data.columns:
        raise KeyError(f"'{target_column}' not found in the dataset columns: {data.columns.tolist()}")

    # Split the data into features and target variable
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Preprocess the features (convert categorical variables to numerical)
    X = pd.get_dummies(X, drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Data Scaling ---
    X_train, X_test = scale_data(X_train, X_test)

    # --- Handle Class Imbalance ---
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

# Define datasets and target columns
datasets = {
    "dt1.csv": "Outcome Variable",
    "dt2.csv": "prognosis"
}

# Run models on each dataset
for dataset, target_column in datasets.items():
    data_path = f'/workspaces/advanced-ml-course/disease-symptoms/data/{dataset}'
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path, target_column)

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
            print(f"{model_name} Accuracy on {dataset}:", accuracy)
            reports[model_name] = report

        # Print combined classification report for the dataset
        print_combined_report(reports, dataset)
    except KeyError as e:
        print(f"Error processing {dataset}: {e}")
