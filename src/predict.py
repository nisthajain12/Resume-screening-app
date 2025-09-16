import joblib
from preprocess import clean_text

def load_artifacts():
    model = joblib.load("models/resume_model.pkl")
    tfidf = joblib.load("models/tfidf.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return model, tfidf, le

def predict_resume(text):
    model, tfidf, le = load_artifacts()
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    return le.inverse_transform([pred])[0]
