import joblib

def save_model(model, filepath, feature_names=None):
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    joblib.dump(model_data, filepath)

def load_model(filepath):
    model_data = joblib.load(filepath)
    return model_data['model'], model_data.get('feature_names', None)