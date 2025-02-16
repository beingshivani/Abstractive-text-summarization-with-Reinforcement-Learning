from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize(dataset_name="cnn_dailymail", model_name="facebook/bart-large-cnn", split="train"):
    dataset = load_dataset(dataset_name, "3.0.0", split=split)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        inputs = examples["article"]
        targets = examples["highlights"]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset, tokenizer

if __name__ == "__main__":
    dataset, tokenizer = load_and_tokenize(split="train[:1%]")
    print("Sample tokenized input:", dataset[0])
