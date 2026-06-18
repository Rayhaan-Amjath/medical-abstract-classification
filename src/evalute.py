from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer

from transformers import Trainer
import numpy as np
from sklearn.metrics import classification_report

MODEL_PATH = "models/distilbert_medical"

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)


trainer = Trainer(
    model=model
)

predictions = trainer.predict(tokenized_test)

y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Neoplasms",
            "Digestive System Diseases",
            "Nervous System Diseases",
            "Cardiovascular Diseases",
            "General Pathological Conditions"
        ]
    )
)