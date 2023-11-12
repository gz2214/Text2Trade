# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 23:37:34 2023

@author: dinos
"""

import json
import pytz
import pandas as pd 
from datetime import datetime, timedelta
import datasets
from transformers import AutoTokenizer
f = open('../data/business_news.json')
business_news=json.load(f)
f1=open('../data/tech_news.json')
tech_news=json.load(f1)

total_items=len(business_news)
business_items_per_part=total_items//10
for i in range(10):
    start_index = i * business_items_per_part
    end_index = (i + 1) * business_items_per_part if i < 9 else total_items

    part_data = business_news[start_index:end_index]

    # Save each part into separate JSON files
    with open(f'../data/business_news_{i + 1}.json', 'w') as part_file:
        json.dump(part_data, part_file, indent=4)


total_items=len(tech_news)
tech_items_per_part=total_items//10
for i in range(10):
    start_index = i *tech_items_per_part
    end_index = (i + 1) * tech_items_per_part if i < 9 else total_items

    part_data = business_news[start_index:end_index]

    # Save each part into separate JSON files
    with open(f'../data/tech_news_{i + 1}.json', 'w') as part_file:
        json.dump(part_data, part_file, indent=4)