import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\S+@\S+', ' ', text)           # remove emails
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # remove urls
    text = re.sub(r'[^a-zA-Z]', ' ', text)         # keep only letters
    text = text.lower()
    tokens = [w for w in text.split() if w not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)
