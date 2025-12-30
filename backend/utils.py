import torch
import spacy
import pickle
from huggingface_hub import hf_hub_download
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification

class TextClassifier:
    def __init__(self, model_path, token):
        # Load model and tokenizer
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            token=token,
            device_map = None
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            model_path,
            token=token
        )

        # Load label encoder
        LABEL_ENCODER_FILENAME = "label_encoder.pkl"
        LOCAL_ENCODER_FILE_PATH = Path(model_path + "/" + LABEL_ENCODER_FILENAME)

        if LOCAL_ENCODER_FILE_PATH.exists():
            # Local development
            with open(LOCAL_ENCODER_FILE_PATH, "rb") as f:
                self.label_encoder = pickle.load(f)
        else:
            # Streamlit Cloud
            downloaded_path = hf_hub_download(
                repo_id=model_path,
                filename=LABEL_ENCODER_FILENAME
            )
            with open(downloaded_path, "rb") as f:
                self.label_encoder = pickle.load(f)
        
        # Set Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
       
        self.model.eval()

        # Load spacy
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess_text(self, text):
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lower case
        doc = self.nlp(str(text).lower())

        tokens = [token.lemma_ for token in doc 
                    if not token.is_stop 
                    and not token.is_punct 
                    and not token.is_space
                    and token.text.strip()]

        return " ".join(tokens)
    
    def predict(self, text, top_k=3):

        # Preprocess text
        processed_text = self.preprocess_text(text)

        if not processed_text:
            return {
                'error': 'Text is empty after preprocessing',
                'predicted_class': None,
                'confidence': 0.0,
                'all_predictions': []
            }
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt'
        )

        # Move to Device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

        # Get Top K Predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, len(self.label_encoder.classes_)))

        # Format results
        predicted_idx = top_indices[0].cpu().item()
        predicted_class = self.label_encoder.classes_[predicted_idx]
        confidence = top_probs[0].cpu().item()

        # all predictions
        all_predictions = [{
                'class': self.label_encoder.classes_[idx.cpu().item()],
                'probability': prob.cpu().item()
            }
            for idx, prob in zip(top_indices, top_probs)
        ]

        return {
            'original_text': text,
            'processed_text': processed_text,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        }
