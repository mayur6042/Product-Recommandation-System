import os
import joblib
from typing import List


def load_model(model_dir='models'):
    clf_path = os.path.join(model_dir, 'sentiment_model.pkl')
    vect_path = os.path.join(model_dir, 'vectorizer.pkl')
    clf = joblib.load(clf_path)
    vect = joblib.load(vect_path)
    return clf, vect


def predict_text(texts: List[str], model_dir='models'):
    clf, vect = load_model(model_dir)
    X = vect.transform(texts)
    return clf.predict(X)


if __name__ == '__main__':
    sample = ["This product is great!", "Terrible, broke after one use."]
    print(predict_text(sample))
