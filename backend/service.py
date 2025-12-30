from backend.model_loader import get_classifier

def get_class_metadata():
    classifier = get_classifier()

    classes = classifier.label_encoder.classes_.tolist()
    return {
        "classes": classes,
        "num_classes": len(classes)
    }

def predict_text(payload: dict):
    classifier = get_classifier()

    try:
        result = classifier.predict(
            payload["text"],
            top_k=payload.get("top_k")
        )
        return result
    except Exception as e:
        return {"error": str(e)}