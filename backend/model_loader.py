from backend.utils import TextClassifier
import os

classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        model_path = os.getenv("MODEL_PATH")
        token = os.getenv("HF_TOKEN")
        classifier = TextClassifier(model_path, token)
    return classifier