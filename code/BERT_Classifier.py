import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import ast
import re
import os
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
import torch.nn as nn

#function to split the text into chunks of tokens - ensure each chunk is less than 512 tokens
def chunk_tokens(text, max_tokens=512):
    tokens = tokenizer.tokenize(text) #tokenize text
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        current_chunk.append(token) #append token to current chunk
        current_length += 1

        #once the limit is reached, the current chunk is saved, and a new chunk begins
        if current_length >= max_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

#process a block of data and create df_train and df_test
def process_blocks(block, block_name):
    train_dataset = block['train']
    test_dataset = block['test']

    train_dates, train_titles, train_labels = [], [], []
    test_dates, test_titles, test_labels = [], [], []

    for item in tqdm(train_dataset, desc=f'Processing Train Data ({block_name})'):
        date = item['Date']
        title = item['Title']
        label = item['Label']
        #tokenizes the titles into chunks
        chunks = chunk_tokens(title)

        for chunk in chunks:
            train_dates.append(date)
            train_titles.append(' '.join(chunk))
            train_labels.append(label)

    for item in tqdm(test_dataset, desc=f'Processing Test Data ({block_name})'):
        date = item['Date']
        title = item['Title']
        label = item['Label']
        chunks = chunk_tokens(title)

        for chunk in chunks:
            test_dates.append(date)
            test_titles.append(' '.join(chunk))
            test_labels.append(label)

    df_train = pd.DataFrame({'Date': train_dates, 'Title': train_titles, 'Label': train_labels})
    df_test = pd.DataFrame({'Date': test_dates, 'Title': test_titles, 'Label': test_labels})

    return df_train, df_test

def load_train_data(blocks, tokenizer, max_tokens=512, batch_size=16, shuffle=True, train_ratio=0.8):
    # Create empty lists to store data from blocks
    titles = []
    labels = []

    # Merge data from blocks 0 to 3
    for block in blocks[:4]:  # Blocks 0 to 3
        # Extract titles and labels
        block_titles = block['train']['Title']
        block_labels = block['train']['Label']

        # Append titles and labels
        titles.extend(block_titles)
        labels.extend(block_labels)

    # Merge data from block4['train']
    block4_train_titles = blocks[4]['train']['Title']
    block4_train_labels = blocks[4]['train']['Label']

    # Append titles and labels
    titles.extend(block4_train_titles)
    labels.extend(block4_train_labels)

    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)

    # Tokenize and process titles
    input_ids_list = []
    attention_masks_list = []

    for title in titles:
        inputs = tokenizer(title, return_tensors='pt', padding=True, truncation=True, max_length=max_tokens)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)

    # Find the maximum sequence length
    max_sequence_length = max(input_ids.shape[1] for input_ids in input_ids_list)

    # Pad input_ids and attention_mask tensors to have the same length
    for i in range(len(input_ids_list)):
        input_ids_list[i] = torch.cat([input_ids_list[i], torch.zeros(1, max_sequence_length - input_ids_list[i].shape[1], dtype=torch.long)], dim=1)
        attention_masks_list[i] = torch.cat([attention_masks_list[i], torch.zeros(1, max_sequence_length - attention_masks_list[i].shape[1], dtype=torch.long)], dim=1)

    # Stack input_ids tensors and attention_mask tensors along dim=0
    input_ids = torch.stack(input_ids_list)
    attention_masks = torch.stack(attention_masks_list)

    # Split data into train and validation sets based on train_ratio
    train_size = int(train_ratio * len(labels))
    val_size = len(labels) - train_size

    train_dataset = TensorDataset(input_ids[:train_size], attention_masks[:train_size], labels[:train_size])
    val_dataset = TensorDataset(input_ids[train_size:], attention_masks[train_size:], labels[train_size:])

    # Create DataLoader for train and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

def load_data_for_testing(blocks, tokenizer, max_tokens=512, batch_size=16, shuffle=True):
    # Create empty lists to store data
    titles = []
    labels = []
    input_ids_list = []  # Add this line
    attention_masks_list = []  # Add this line

    # Merge data from blocks 0 to 4 for training
    for block in blocks[:5]:  # Blocks 0 to 4
        # Extract titles and labels
        block_titles = block['train']['Title']
        block_labels = block['train']['Label']

        # Append titles and labels
        titles.extend(block_titles)
        labels.extend(block_labels)

    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)

    # Tokenize and process titles 
    for title in titles:
        inputs = tokenizer(title, return_tensors='pt', padding=True, truncation=True, max_length=max_tokens)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)

    # Find the maximum sequence length
    max_sequence_length = max(input_ids.shape[1] for input_ids in input_ids_list)

    # Pad input_ids and attention_mask tensors to have the same length
    for i in range(len(input_ids_list)):
        input_ids_list[i] = torch.cat([input_ids_list[i], torch.zeros(1, max_sequence_length - input_ids_list[i].shape[1], dtype=torch.long)], dim=1)
        attention_masks_list[i] = torch.cat([attention_masks_list[i], torch.zeros(1, max_sequence_length - attention_masks_list[i].shape[1], dtype=torch.long)], dim=1)

    # Stack input_ids tensors and attention_mask tensors along dim=0
    input_ids = torch.stack(input_ids_list)
    attention_masks = torch.stack(attention_masks_list)

    # Create DataLoader for training data
    train_dataset = TensorDataset(input_ids, attention_masks, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # Extract day-level labels and titles for testing
    test_dates = blocks[4]['test']['Date']
    test_titles = blocks[4]['test']['Title']
    test_labels = blocks[4]['test']['Label']

    # Tokenize and process titles for testing
    test_input_ids_list = []
    test_attention_masks_list = []

    for title in test_titles:
        inputs = tokenizer(title, return_tensors='pt', padding=True, truncation=True, max_length=max_tokens)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        test_input_ids_list.append(input_ids)
        test_attention_masks_list.append(attention_mask)

    # Find the maximum sequence length for testing data
    max_test_sequence_length = max(input_ids.shape[1] for input_ids in test_input_ids_list)

    # Pad input_ids and attention_mask tensors for testing data to have the same length
    for i in range(len(test_input_ids_list)):
        test_input_ids_list[i] = torch.cat([test_input_ids_list[i], torch.zeros(1, max_test_sequence_length - test_input_ids_list[i].shape[1], dtype=torch.long)], dim=1)
        test_attention_masks_list[i] = torch.cat([test_attention_masks_list[i], torch.zeros(1, max_test_sequence_length - test_attention_masks_list[i].shape[1], dtype=torch.long)], dim=1)

    # Stack input_ids tensors and attention_mask tensors for testing data along dim=0
    test_input_ids = torch.stack(test_input_ids_list)
    test_attention_masks = torch.stack(test_attention_masks_list)

    # Convert labels for testing data to a tensor
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create DataLoader for testing data
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def calculate_num_chunks_per_day(dataset):
    # Convert the dataset to a DataFrame
    df = pd.DataFrame(dataset)

    # Group by unique "Date" and count the number of chunks (labels) per day
    num_chunks_per_day = df.groupby("Date")["Label"].count().values

    return num_chunks_per_day

def main():
    dataset_path_block0 = '../data/block0'
    dataset_path_block1 = '../data/block1'
    dataset_path_block2 = '../data/block2'
    dataset_path_block3 = '../data/block3'
    dataset_path_block4 = '../data/block4'

    block0 = load_from_disk(dataset_path_block0)
    block1 = load_from_disk(dataset_path_block1)
    block2 = load_from_disk(dataset_path_block2)
    block3 = load_from_disk(dataset_path_block3)
    block4 = load_from_disk(dataset_path_block4)

    blocks = [block0, block1, block2, block3, block4]
    block_names = ['block0', 'block1', 'block2', 'block3', 'block4']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=2 # Binary classification (up/down)
                                                          )
    for block, block_name in zip(blocks, block_names):
        df_train, df_test = process_blocks(block, block_name)
        df_train.to_csv(f'../data/data/{block_name}_train.csv', index=False)
        df_test.to_csv(f'../data/data/{block_name}_test.csv', index=False)

    train_dataloader, val_dataloader = load_train_data(blocks, tokenizer, batch_size=16, shuffle=True)

    # Define the optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    #we want to leave the last block as "test" after fine tuning.
    epochs = 30
    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    for epoch in range(epochs):
    # ========================================
    #               Training
    # ========================================

    print(f'Epoch {epoch + 1}/{epochs}')
    print('Training...')

    avg_train_loss=[]
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
        print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        b_input_ids = batch[0].squeeze(1)
        b_input_mask = batch[1]
        b_labels = batch[2]

        #print("Shape of input_ids:", b_input_ids.shape)

        model.zero_grad()
        outputs = model(input_ids = b_input_ids, attention_mask=b_input_mask)

        #loss = outputs.loss
        loss = nn.CrossEntropyLoss()(outputs.logits, b_labels)
        logits = outputs.logits
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print("Average training loss: {0:.2f}".format(avg_train_loss))

    train_dataloader, test_dataloader = load_data_for_testing(blocks, tokenizer, batch_size=16, shuffle=True)

    block4_test = pd.read_csv("../data/block4_test.csv")  # Replace "path_to_block4_test.csv" with the actual file path
    num_chunks_per_day = calculate_num_chunks_per_day(block4_test)

    day_predictions = [] #store predictions (logits) for each chunk of text
    day_probabilities = [] #store probability of class 1 for each chunk
    day_ground_truth_labels = [] #store ground truth labels for each chunk
    num_examples_per_day = None

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for batch in test_dataloader:
            #extract inputs for a day (chunks of text)
            day_input_ids = batch[0].squeeze(1)
            day_attention_mask = batch[1]
            day_labels = batch[2]

            outputs = model(input_ids=day_input_ids, attention_mask=day_attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

            #aggregate predictions for the day
            day_predictions.append(torch.argmax(logits, dim=1))
            day_probabilities.append(probabilities[:, 1])  # Probability of class 1

            # Ground truth labels for the day
            day_ground_truth_labels.append(day_labels)

    day_predictions = torch.cat(day_predictions)
    print("Shape of day_predictions:", day_predictions.shape)
    day_probabilities = torch.cat(day_probabilities)
    day_ground_truth_labels = torch.cat(day_ground_truth_labels)

    #majority vote for each day
    day_final_labels = []
    current_index = 0 

    for num_chunks in num_chunks_per_day:
        day_predictions_i = day_predictions[current_index:current_index + num_chunks]
        day_final_label = 1 if (day_predictions_i.sum() > (num_chunks / 2)) else 0
        day_final_labels.extend([day_final_label] * num_chunks)
        current_index += num_chunks

    #probability of class 1 for each day
    day_probabilities = [day_probabilities[i:i + num_chunks] for i, num_chunks in enumerate(num_chunks_per_day)]
    day_avg_probabilities = [torch.mean(prob) for prob in day_probabilities]

    #ensure that day_final_labels and day_ground_truth_labels have the same length
    day_final_labels = day_final_labels[:len(day_ground_truth_labels)]

    # Calculate evaluation metrics
    accuracy = accuracy_score(day_ground_truth_labels, day_final_labels)
    f1 = f1_score(day_ground_truth_labels, day_final_labels)
    precision = precision_score(day_ground_truth_labels, day_final_labels)
    map = average_precision_score(day_ground_truth_labels, day_final_labels)

    #print("Length of day_ground_truth_labels:", len(day_ground_truth_labels))
    #print("Length of day_final_labels:", len(day_final_labels))
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Mean Average Precision: {map:.2f}")

    metrics_dict = {
        "Precision": precision,
        "Mean Average Precision": map,
        "F1 Score": f1,
        "Accuracy": accuracy
    }

    file_name = "../results/BERT_classifier_eval.txt"

    with open(file_name, 'w') as file:
        for metric_name, metric_value in metrics_dict.items():
            file.write(f"{metric_name}: {metric_value:.6f}\n")

    print(f"Metrics saved to {file_name}")

    for i, label in enumerate(day_final_labels):
        print(f"Day {i + 1} Aggregated Label: {label}")

    output_file_name = "../results/BERT_classifier_daylevel_class.txt"

    # Open the file for writing
    with open(output_file_name, 'w') as output_file:
        for i, label in enumerate(day_final_labels):
            output_file.write(f"Day {i + 1} Aggregated Label: {label}\n")

    # Print a message to confirm that the file has been saved
    print(f"Day final labels saved to {output_file_name}")

    for i, prob in enumerate(day_avg_probabilities):
        print(f"Day {i + 1} Average Probability of 1: {prob:.2f}")

    # Specify the file name for the average probabilities
    average_probabilities_file_name = "../results/BERT_classifier_daylevel_prob.txt"

    # Open the file for writing
    with open(average_probabilities_file_name, 'w') as file:
        # Write each day's average probability in the desired format
        for i, prob in enumerate(day_avg_probabilities):
            file.write(f"Day {i + 1} Average Probability of 1: {prob:.2f}\n")

    # Print a message to confirm that the file has been saved
    print(f"Average probabilities saved to {average_probabilities_file_name}")

if __name__ == '__main__':
    main()
