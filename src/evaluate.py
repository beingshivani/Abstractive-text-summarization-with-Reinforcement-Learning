from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def generate_summary(model, tokenizer, text, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    summary_ids = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_baseline = AutoModelForSeq2SeqLM.from_pretrained("./rl_summarizer_baseline")
    model_rl = AutoModelForSeq2SeqLM.from_pretrained("./rl_summarizer_rl_finetuned")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    sample_text = "Just drop your sample article text here. A few sentences should do the trick and show what kind of data we're talking about."
    
    print("Baseline Summary:")
    print(generate_summary(model_baseline, tokenizer, sample_text))
    
    print("\nRL-Finetuned Summary:")
    print(generate_summary(model_rl, tokenizer, sample_text))

