def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    import joblib
    return joblib.load(filename)