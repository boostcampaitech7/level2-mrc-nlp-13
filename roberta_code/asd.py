import argparse
from tqdm import tqdm
import pandas as pd
import torch
import re
import json
from datasets import Dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import string
from collections import Counter

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate QA model")
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation")
    return parser.parse_args()

def load_and_preprocess_data(dataset_path):
    dataset = load_from_disk(dataset_path)
    
    def extract_data(split):
        return [
            {
                'id': example['id'],
                'context': example['context'],
                'question': example['question'],
                'answer': example['answers']['text'][0] if example['answers']['text'] else "",
                'answer_start': example['answers']['answer_start'][0] if example['answers']['answer_start'] else 0,
                'document_id': example['document_id'],
                'title': example['title']
            }
            for example in tqdm(dataset[split], desc=f"Preprocessing {split} data")
        ]
    
    train_data = extract_data('train')
    valid_data = extract_data('validation')
    
    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    
    for df in [train_df, valid_df]:
        df['text'] = 'Context: ' + df['context'] + '\nQuestion: ' + df['question'] + '\nAnswer: ' + df['answer']
    
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    return train_dataset, valid_dataset

def setup_model_and_tokenizer(model_id):
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config=bnb_config,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16)
    
    if model is None:
        raise ValueError(f"Failed to load model with ID: {model_id}")
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, valid_dataset):
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    training_args = TrainingArguments(
        output_dir='model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        fp16=True, 
        seed=2024,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        dataset_text_field='text',
        peft_config=peft_config,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
    )
    
    trainer.train()
    trainer.save_model()
    
    return trainer

def generate_answer(model, tokenizer, context, question):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    answer_match = re.search(r'Answer:\s*(.*?)(?:\n|$)', result, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = ""
    
    # Ensure the answer is within the context
    if answer not in context:
        sentences = context.split('.')
        best_sentence = max(sentences, key=lambda s: len(set(s.lower().split()) & set(answer.lower().split())))
        answer = best_sentence.strip()
    
    return answer

def normalize_answer(s):
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    return ' '.join(remove_punc(s.lower()).split())

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def evaluate_model(model, tokenizer, valid_dataset):
    predictions = {}
    references = []
    
    for i, example in enumerate(tqdm(valid_dataset, desc="Evaluating")):
        try:
            pred = generate_answer(model, tokenizer, example['context'], example['question'])
            predictions[example['id']] = pred
            references.append({
                'id': example['id'],
                'answer': {'text': example['answer']}
            })
        except Exception as e:
            print(f"Error processing example {example['id']}: {str(e)}")
    
    exact_match = sum(exact_match_score(predictions[ref['id']], ref['answer']['text']) for ref in references) / len(references)
    f1 = sum(f1_score(predictions[ref['id']], ref['answer']['text']) for ref in references) / len(references)
    
    eval_results = {
        "eval_samples": len(predictions),
        "exact_match": exact_match * 100,
        "f1": f1 * 100
    }
    
    with open('all_result.json', 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)
    
    with open('prediction.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    
    return eval_results

def main():
    args = parse_arguments()
    
    try:
        dataset_path = '../data/train_dataset'
        model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
        
        train_dataset, valid_dataset = load_and_preprocess_data(dataset_path)
        model, tokenizer = setup_model_and_tokenizer(model_id)
        
        if args.do_train:
            print("Starting training...")
            trainer = train_model(model, tokenizer, train_dataset, valid_dataset)
            print("Training completed.")
        
        if args.do_eval:
            print("Starting evaluation...")
            evaluation_results = evaluate_model(model, tokenizer, valid_dataset)
            print("Evaluation results:", evaluation_results)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()