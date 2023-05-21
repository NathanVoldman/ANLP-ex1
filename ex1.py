from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed, pipeline
from datasets import load_dataset, load_metric
from collections import defaultdict
import wandb
import numpy as np
import os
import sys
import time

def cut_dataset(dataset, num_samples):
    num_samples = len(dataset) if num_samples == -1 else num_samples
    return dataset.select([i for i in range(num_samples)])


def prepare_datasets(num_train, num_val, num_test):
    sst2 = load_dataset("sst2")
    train_set = cut_dataset(sst2['train'], num_train)
    val_set = cut_dataset(sst2['validation'], num_val)
    test_set = cut_dataset(sst2['test'], num_test)

    return train_set, val_set, test_set


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(
        predictions=predictions, references=labels)["accuracy"]
    return {"accuracy": accuracy}


def get_tokenize_function(tokenizer):
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    return preprocess_function


def get_trainer(model, tokenizer, train_set, val_set, rand_seed, model_name):
    preprocess_function = get_tokenize_function(tokenizer)
    tokenized_train = train_set.map(preprocess_function, batched=True)
    tokenized_val = val_set.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=".",
        report_to="wandb",
        save_strategy="no",
        data_seed=rand_seed,
        run_name=f'{model_name}-{rand_seed}'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer


def finetune_model(trainer):
    train_output = trainer.train()
    return train_output.metrics


def evaluate_model(trainer):
    val_metrics = trainer.evaluate()
    return val_metrics


def get_test_pred(model, tokenizer, test_set):
    classifier = pipeline('sentiment-analysis', model=model,
                          tokenizer=tokenizer, device=0)
    
    start_time = time.time()
    predictions = classifier(test_set['sentence'], padding=False)
    end_time = time.time()
    prediction_time = end_time - start_time

    predictions = [prediction['label'][-1] for prediction in predictions]
    predictions_text_list = [f'{sentence}###{prediction}' for sentence,
                             prediction in zip(test_set['sentence'], predictions)]
    predictions_text = '\n'.join(predictions_text_list)

    with open('predictions.txt', 'w') as pred_writer:
        pred_writer.write(predictions_text)

    return prediction_time

def main():
    args = sys.argv[1:]
    num_seeds = int(args[0])
    num_train = int(args[1])
    num_val = int(args[2])
    num_test = int(args[3])

    train_set, val_set, test_set = prepare_datasets(
        num_train, num_val, num_test)

    model_names = ['bert-base-uncased', 'roberta-base',
                   'google/electra-base-generator']

    # model_name => list of (acc, model, tokenizer) where the index is also the seed
    model_details = defaultdict(list)
    train_time = 0
    res = ""

    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for rand_seed in range(num_seeds):
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2)

            trainer = get_trainer(
                model, tokenizer, train_set, val_set, rand_seed, model_name)
            train_metrics = finetune_model(trainer)
            val_metrics = evaluate_model(trainer)
            wandb.finish()
            # get_preds_from_model(trainer, test_set)

            curr_acc = val_metrics['eval_accuracy']
            model_details[model_name].append((curr_acc, model, tokenizer))
            train_time += train_metrics['train_runtime']

        model_acc_list = [details[0] for details in model_details[model_name]]
        res += f'{model_name},{np.mean(model_acc_list)} +- {np.std(model_acc_list)}\n'

    res += '----\n'
    res += f'train_time,{train_time}\n'

    # model name -> mean model accuracy
    model_mean_acc = {key: np.mean(
        [details[0] for details in value]) for key, value in model_details.items()}
    best_model_name = max(model_mean_acc, key=model_mean_acc.get)

    # list of size number of seed: (acc, model, tokenizer)
    best_model_details = model_details[best_model_name]
    best_seed = np.argmax([details[0] for details in best_model_details])

    # (acc, model, tokenizer)
    best_seed_details = best_model_details[best_seed]
    best_acc = best_seed_details[0]
    print(
        f'best model is: {best_model_name}, best seed is: {best_seed}, best accuracy of chosen model is: {best_acc}')

    best_model = best_seed_details[1]
    best_tokenizer = best_seed_details[2]

    prediction_time = get_test_pred(best_model, best_tokenizer, test_set)
    res += f'predict_time,{prediction_time}'

    with open('res.txt', 'w') as res_writer:
        res_writer.write(res)




if __name__ == "__main__":
    os.environ['HF_DATASETS_CACHE'] = './hf_cache'
    os.environ['HF_HOME'] = './hf_home'
    os.environ['WANDB_PROJECT'] = "ANLP-ex1"
    wandb.login()
    main()
