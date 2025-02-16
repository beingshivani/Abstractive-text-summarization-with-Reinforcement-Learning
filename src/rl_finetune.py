
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from data_loader import load_and_tokenize
from rouge_score import rouge_scorer

def compute_reward(generated, reference):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores["rougeL"].fmeasure

def rl_finetune(model, tokenizer, dataset, num_epochs=1, learning_rate=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for sample in dataset:

            inputs = tokenizer(sample["article"], return_tensors="pt", truncation=True, padding="max_length", max_length=512)

            with tokenizer.as_target_tokenizer():
                reference = tokenizer.decode(sample["labels"], skip_special_tokens=True)

            baseline_ids = model.generate(**inputs, max_length=128)
            baseline_summary = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)
            baseline_reward = compute_reward(baseline_summary, reference)
            
            sampled_ids = model.generate(**inputs, max_length=128, do_sample=True, top_k=50)
            sampled_summary = tokenizer.decode(sampled_ids[0], skip_special_tokens=True)
            sampled_reward = compute_reward(sampled_summary, reference)
            
            reward = sampled_reward - baseline_reward
            
            outputs = model(**inputs, labels=sampled_ids)
            loss = outputs.loss
            rl_loss = loss * reward
            
            optimizer.zero_grad()
            rl_loss.backward()
            optimizer.step()
            
            total_loss += rl_loss.item()
        print(f"Epoch {epoch+1} RL Loss: {total_loss/len(dataset):.4f}")
    return model

if __name__ == "__main__":
    model_name = "facebook/bart-large-cnn"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset, _ = load_and_tokenize(split="train[:1%]")   # it load only 1
    model = rl_finetune(model, tokenizer, dataset, num_epochs=1)
    model.save_pretrained("./rl_summarizer_rl_finetuned")
    tokenizer.save_pretrained("./rl_summarizer_rl_finetuned")

'''
The following code process one sample only at a time
'''
