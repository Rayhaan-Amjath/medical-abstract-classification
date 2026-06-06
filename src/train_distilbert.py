import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import evaluate 
import numpy as np

df = pd.read_csv(
    'data/Medical-Abstracts-TC-Corpus-main/medical_tc_train.csv'
    )

df["label"] = df["condition_label"] - 1 #hugging face models prefer to start from 0, this just offsets it

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Convert to HF datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["medical_abstract"],
        truncation=True,
        padding="max_length",
        max_length=384 #updated because average length of abstracts was around 277
    )
#MAX od 256 tokens so it has a fixed context window, truncation is on, meaning if an abstract is longer 
#we cut if off, and padding is on so if the abstract is too short, it adds padding 

#tokenize dataset
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_test = test_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_train = tokenized_train.remove_columns(
    ["condition_label", "medical_abstract", "__index_level_0__"]
) #model does not care about these columns

tokenized_test = tokenized_test.remove_columns(
    ["condition_label", "medical_abstract", "__index_level_0__"]
)

#convert data into pytorhc tensors 
tokenized_train.set_format("torch") 
tokenized_test.set_format("torch")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=5
) #num labels = 5 bc we have 5 categories 

lengths = []

for text in df["medical_abstract"]:
    lengths.append(
        len(tokenizer.tokenize(text))
    )

print("Average length:", sum(lengths)/len(lengths))
print("Max length:", max(lengths))

