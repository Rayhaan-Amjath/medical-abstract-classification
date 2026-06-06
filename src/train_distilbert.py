import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import evaluate 
import accelerate
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

#metric function - want to measure if our accuracy can beat 55% from logistic regression model 
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred): 
    logits, labels = eval_pred 
    predictions = np.argmax(
        logits, 
        axis=-1
    )
    return accuracy_metric.compute(
        predictions=predictions, 
        references=labels
    )

#Training arguments 
training_args = TrainingArguments(
    output_dir = "./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    
    learning_rate = 2e-5,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True
)

#Trainer 
trainer = Trainer(
    model = model, 
    args = training_args, 
    train_dataset = tokenized_train, 
    eval_dataset = tokenized_test, 
    compute_metrics = compute_metrics
)

