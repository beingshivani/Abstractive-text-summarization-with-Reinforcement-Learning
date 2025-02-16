
from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from data_loader import load_and_tokenize

def fine_tune_model(model_name="facebook/bart-large-cnn", output_dir="./rl_summarizer_baseline"):
    dataset, tokenizer = load_and_tokenize(split="train[:5%]")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
      
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer

if __name__ == "__main__":
    fine_tune_model()
