# Get dataset from [Kaggle](https://www.kaggle.com/competitions/home-data-for-ml-course/overview)

import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def setup_kaggle_api():
    """Initialize Kaggle API authentication."""
    # Look for kaggle.json in the default location or environment variable
    kaggle_default_path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    kaggle_config_dir = os.getenv("KAGGLE_CONFIG_DIR")

    if kaggle_config_dir:
        kaggle_json_path = os.path.join(kaggle_config_dir, "kaggle.json")
    else:
        kaggle_json_path = kaggle_default_path

    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(
            f"Could not find kaggle.json file. Please place it in '{kaggle_json_path}', "
            "or set the KAGGLE_CONFIG_DIR environment variable to the folder containing kaggle.json."
        )

    # Authenticate with Kaggle API
    api = KaggleApi()
    api.authenticate()
    return api


def download_dataset(api, competition_name='home-data-for-ml-course'):
    """Download dataset files from Kaggle competition."""
    api.competition_download_files(competition_name, path='./data')
    # Extract downloaded zip file
    import zipfile
    with zipfile.ZipFile('./data/home-data-for-ml-course.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')


def load_dataset():
    """Load training and test datasets."""
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    return train_df, test_df


def main():
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)

    try:
        # Setup and download
        api = setup_kaggle_api()
        download_dataset(api)

        # Load and inspect data
        train_df, test_df = load_dataset()
        print(f"Training set shape: {train_df.shape}")
        print(f"Test set shape: {test_df.shape}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please follow the setup instructions for the Kaggle API: "
              "https://github.com/Kaggle/kaggle-api#api-credentials")


if __name__ == "__main__":
    main()
