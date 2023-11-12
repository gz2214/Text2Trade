# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:02:47 2023

@author: dinos
"""

import json
import pytz
import pandas as pd 
from datetime import datetime, timedelta
import datasets
from datasets import load_from_disk
from transformers import AutoTokenizer
import os

#concatenate data together
all_news=[]
for filename in os.listdir('../data/'):
    print(filename)
    if filename.endswith('.json'):
        file_path = os.path.join('../data/', filename)
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                    # Merge the contents of the JSON file
                all_news=all_news+data
            except json.JSONDecodeError:
                print(f"Error reading {filename}. It's not valid JSON.")
#stock_data
stock_data=pd.read_csv('../data/daily_price_movement.csv')

## change article time to EST
def changeTimeEst(d):
    time=datetime.strptime(d['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=pytz.UTC)
    d['publishedAt']=time.astimezone(pytz.timezone('US/Eastern'))
    return(d)

#function to concatenate headers for stock date 
def articles_concat(d):
    print(d)
    ind=stock_data[stock_data['date']==d].index[0]
    if ind!=0:
        articles=[a for a in all_news if a['publishedAt']>=stock_data['date'][ind-1] and a['publishedAt']<=d]
    else:
        articles=[a for a in all_news if a['publishedAt']>=(d-timedelta(days=3)) and a['publishedAt']<=d]   
    headers=[a['title'] for a in articles]
    label=stock_data['price_movement'].loc[ind]
    dic={'Date':d,'Title':headers,'Label':label}
    return(dic)

def blocks(my_list,num_parts):
    block_size, remainder = divmod(len(my_list), num_parts)
    blocks = [my_list[i * block_size + min(i, remainder):(i + 1) * block_size + min(i + 1, remainder)] for i in range(num_parts)]
    return(blocks)

def train_split(pct,l):
    splt=round(len(l)*.8)
    train=l[:splt]
    test=l[splt:]
    return({'train':train,'test':test})
def dataSet(data):
    train=pd.DataFrame.from_dict(data['train'])
    test=pd.DataFrame.from_dict(data['test'])
    train['Title']=train['Title'].apply(lambda x: ' '.join(x))
    test['Title']=test['Title'].apply(lambda x: ' '.join(x))
    ds_train=datasets.Dataset.from_pandas(train)
    ds_test=datasets.Dataset.from_pandas(test)
    x=datasets.DatasetDict({'train':ds_train, 'test':ds_test})
    return(x)

def tokenize_function(examples):
    return tokenizer(examples["Title"], padding="max_length", truncation=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

##actual preprocessing
#change timezone in articles into est
list(map(changeTimeEst, all_news))
est = pytz.timezone('US/Eastern')
#change timezone in stock data into EST
stock_data.date=stock_data.date.apply(lambda x:  datetime.strptime(x, '%Y-%m-%d').replace(hour=9, minute=0, second=0))
stock_data.date=stock_data['date'].apply(lambda x: est.localize(x))
#concenate article titles for each date 
plsTry=[articles_concat(d) for d in stock_data['date']]
plsTry1=blocks(plsTry,5)
data=[train_split(.8,p) for p in plsTry1]

#create datasets 
processed_data={}

for i,dat in enumerate(data):
    ds=dataSet(dat)
    processed_data['block'+str(i)]=dataSet(dat)


for i in processed_data.keys():
    processed_data[i].save_to_disk('../data/'+i)
## example for a given block
'''
block0=load_from_disk('../data/block0')
tokenized_dataset=block0.map(tokenize_function, batched=True)
'''