import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import BertTokenizer, BertModel

def compute_embedding(text_list, tokenizer, model):
    '''
    Description: Computes the embedding of a given list of text strings using a pre-defined tokenizer and model (typically from the BERT family). It tokenizes the input texts, processes them through the model to get embeddings, and computes a context vector by averaging the token embeddings, considering the attention mask. The final output is the mean of these context vectors.
    
    Input: text_list (List of strings) - A list of text strings for which embeddings are to be computed.
    Output: Tensor - The mean of the context vectors computed from the embeddings of the input text list.
    '''
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = inputs.attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    context_vector = sum_embeddings / sum_mask
    
    return context_vector.mean(dim=0)

def daily_embedding(chunks, tokenizer, model):
    '''
    Description: Averages embeddings for a batch of text chunks. Iterates over each chunk in the input, computes its embedding using compute_embedding, and then calculates the mean of these embeddings to represent the entire batch.
    
    Input: chunks (List of strings) - A list of text chunks, each of which is a sublist of tokens.
    Output: Tensor - The average embedding representing the entire batch of input text chunks.
    '''
    batch_embeddings = []

    for chunk in chunks:
        batch_embeddings.append(compute_embedding(chunk, tokenizer, model))

    daily_embedding = torch.stack(batch_embeddings).mean(dim=0)
    return daily_embedding

def chunk_tokens(tokens, max_tokens=512):
    '''
    Description: Divides a list of tokens into smaller chunks with a specified maximum length. Ensures each chunk contains a sequence of tokens that does not exceed the maximum token count.
    
    Input: tokens (List of strings) - A list of tokens to be chunked; max_tokens (int, default 512) - Maximum number of tokens allowed in each chunk.
    Output: List of lists - A list where each element is a chunk of tokens, each chunk not exceeding the specified maximum length.
    '''
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        current_chunk.append(token)
        current_length += 1

        if current_length >= max_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def compute_embeddings_for_block(block, tokenizer, model):
    '''
    Description: Processes a 'block' of data containing 'train' and 'test' datasets. For each dataset, it computes date-wise embeddings of titles by tokenizing, chunking, and then computing their daily embeddings. The embeddings and corresponding dates are stored in pandas dataframes for training and testing datasets.
    
    Input: block (Dict) - A dictionary containing 'train' and 'test' datasets.
    Output: Tuple (DataFrame, DataFrame) - Two pandas dataframes, one for training and one for testing datasets, each containing date-wise title embeddings.
    '''
    train_dataset = block['train']
    test_dataset = block['test']

    train_dates = []
    test_dates = []
    train_title_embeddings = []
    test_title_embeddings = []

    for train_dataset in tqdm(train_dataset.select(range(len(train_dataset))), desc='Processing'):
        date = train_dataset['Date']
        titles = train_dataset['Title']
        tokens = tokenizer.tokenize(titles)
        chunks = chunk_tokens(tokens)

        train_dates.append(date)
        train_title_embeddings.append(daily_embedding(chunks, tokenizer, model))

    for test_dataset in tqdm(test_dataset.select(range(len(test_dataset))), desc='Processing'):
        date = test_dataset['Date']
        titles = test_dataset['Title']
        tokens = tokenizer.tokenize(titles)
        chunks = chunk_tokens(tokens)
        
        test_dates.append(date)
        test_title_embeddings.append(daily_embedding(chunks, tokenizer, model))

    df_train = pd.DataFrame({'Date': train_dates, 'Title_Embedding': train_title_embeddings})
    df_test = pd.DataFrame({'Date': test_dates, 'Title_Embedding': test_title_embeddings})
    
    return df_train, df_test

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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    blocks = [block0, block1, block2, block3, block4]
    block_names = ['block0', 'block1', 'block2', 'block3', 'block4']

    for block, block_name in zip(blocks, block_names):
        df_train, df_test = compute_embeddings_for_block(block, tokenizer, model)
        df_train.to_csv(f'../data/{block_name}_train.csv', index=False)
        df_test.to_csv(f'../data/{block_name}_test.csv', index=False)

if __name__ == '__main__':
    main()