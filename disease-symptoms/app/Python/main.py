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
from models.model_handlers import save_model, load_model
from pathlib import Path
import os

# Variable to control whether to show confusion matrix
show_confusion_matrix = False

def save_class_mapping(class_mapping, file_path):
    with open(file_path, 'w') as f:
        for index, class_name in class_mapping.items():
            f.write(f"{index},{class_name}\n")

def random_forest(X_train, X_test, y_train, y_test, class_mapping):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    save_model(clf, f'disease-symptoms/app/Python/models/trained/random_forest_model_{dataset_name}.pkl', feature_names=X_train.columns.tolist())
    save_class_mapping(class_mapping, f'disease-symptoms/app/Python/models/trained/random_forest_class_mapping_{dataset_name}.csv')

    return accuracy, report


def decision_tree(X_train, X_test, y_train, y_test, class_mapping):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    save_model(clf, f'disease-symptoms/app/Python/models/trained/decision_tree_model_{dataset_name}.pkl', feature_names=X_train.columns.tolist())
    save_class_mapping(class_mapping, f'disease-symptoms/app/Python/models/trained/decision_tree_class_mapping_{dataset_name}.csv')
    
    return accuracy, report


def svm(X_train, X_test, y_train, y_test, class_mapping):
    clf = SVC(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    save_model(clf, f'disease-symptoms/app/Python/models/trained/svm_model_{dataset_name}.pkl', feature_names=X_train.columns.tolist())
    save_class_mapping(class_mapping, f'disease-symptoms/app/Python/models/trained/svm_class_mapping_{dataset_name}.csv')
    
    return accuracy, report


def logistic_regression(X_train, X_test, y_train, y_test, class_mapping):
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    save_model(clf, f'disease-symptoms/app/Python/models/trained/logistic_regression_model_{dataset_name}.pkl', feature_names=X_train.columns.tolist())
    save_class_mapping(class_mapping, f'disease-symptoms/app/Python/models/trained/logistic_regression_class_mapping_{dataset_name}.csv')
    
    return accuracy, report


def gradient_boosting(X_train, X_test, y_train, y_test, class_mapping):
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    save_model(clf, f'disease-symptoms/app/Python/models/trained/gradient_boosting_model_{dataset_name}.pkl', feature_names=X_train.columns.tolist())
    save_class_mapping(class_mapping, f'disease-symptoms/app/Python/models/trained/gradient_boosting_class_mapping_{dataset_name}.csv')
    
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
    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)


def load_and_preprocess_data(data_path, target_column):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Ensure the target column exists before processing
    if target_column not in data.columns:
        raise KeyError(
            f"'{target_column}' not found in the dataset columns: {data.columns.tolist()}")

    # Split the data into features and target variable
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Preprocess the features (convert categorical variables to numerical)
    X = pd.get_dummies(X, drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # --- Data Scaling ---
    X_train, X_test = scale_data(X_train, X_test)

    # --- Handle Class Imbalance ---
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Create class mapping
    class_mapping = {index: class_name for index, class_name in enumerate(y.unique())}

    return X_train, X_test, y_train, y_test, class_mapping


# Define datasets and target columns
datasets = {
    "dt1.csv": "Outcome Variable",
    "dt2.csv": "prognosis"
}

# Get the root path of the project
root_path = Path(__file__).parent.parent.parent

# Run models on each dataset
for dataset, target_column in datasets.items():
    global dataset_name
    print(f"\nResults in {dataset}:")

    data_path = root_path / f'data/{dataset}'
    try:
        X_train, X_test, y_train, y_test, class_mapping = load_and_preprocess_data(
            data_path, target_column)

        models = {
            "Random Forest": random_forest,
            "Decision Tree": decision_tree,
            "SVM": svm,
            "Logistic Regression": logistic_regression,
            "Gradient Boosting": gradient_boosting
        }

        reports = {}
        for model_name, model_func in models.items():
            dataset_name = Path(dataset).stem
            accuracy, report = model_func(X_train, X_test, y_train, y_test, class_mapping)
            print(f"{model_name} Accuracy on {dataset}:", accuracy)
            reports[model_name] = report

        # Print combined classification report for the dataset
        if show_confusion_matrix:
            print_combined_report(reports, dataset)

    except KeyError as e:
        print(f"Error processing {dataset}: {e}")
