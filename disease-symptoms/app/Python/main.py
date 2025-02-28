import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from models.model_handlers import save_model
from pathlib import Path
import os

# Variable to control whether to show confusion matrix
show_confusion_matrix = False

def save_class_mapping(class_mapping, file_path):
    with open(file_path, 'w') as f:
        for index, class_name in class_mapping.items():
            f.write(f"{index},{class_name}\n")

def train_and_evaluate_model(clf, X_train, X_test, y_train, y_test, class_mapping, model_name, dataset_name):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    save_model(clf, f'models/trained/{model_name}_model_{dataset_name}.pkl', feature_names=X_train.columns.tolist())
    save_class_mapping(class_mapping, f'models/trained/{model_name}_class_mapping_{dataset_name}.csv')

    return accuracy, report, y_prob

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)

def load_and_preprocess_data(data_path, target_column):
    data = pd.read_csv(data_path)
    if target_column not in data.columns:
        raise KeyError(f"'{target_column}' not found in the dataset columns: {data.columns.tolist()}")

    X = data.drop([target_column], axis=1)
    y = data[target_column]
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = scale_data(X_train, X_test)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    class_mapping = {index: class_name for index, class_name in enumerate(y.unique())}

    return X_train_smote, X_test, y_train_smote, y_test, class_mapping

# Define datasets and target columns
datasets = {"dt2.csv": "prognosis"}

# Get the root path of the project
root_path = Path(__file__).resolve().parent.parent.parent

# Run models on each dataset
for dataset, target_column in datasets.items():
    print(f"\nResults in {dataset}:")
    data_path = root_path / f'data/{dataset}'
    try:
        X_train, X_test, y_train, y_test, class_mapping = load_and_preprocess_data(data_path, target_column)
        models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "svm": SVC(probability=True, random_state=42),
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
            "gradient_boosting": GradientBoostingClassifier(random_state=42)
        }

        reports = {}
        for model_name, clf in models.items():
            dataset_name = Path(dataset).stem
            accuracy, report, y_prob = train_and_evaluate_model(clf, X_train, X_test, y_train, y_test, class_mapping, model_name, dataset_name)
            print(f"{model_name.replace('_', ' ').title()} Accuracy on {dataset}:", accuracy)
            reports[model_name] = report

        # Print combined classification report for the dataset
        if show_confusion_matrix:
            print_combined_report(reports, dataset)

    except KeyError as e:
        print(f"Error processing {dataset}: {e}")
