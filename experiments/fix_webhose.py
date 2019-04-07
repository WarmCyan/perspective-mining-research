#!/bin/python3

import os
import json
from tqdm import tqdm

import csv

output_path = "../data/raw/webhose_political/news.csv"

if not os.path.exists(output_path):
    os.makedirs(output_path)
    
with open(output_path, mode='w') as csv_file:
    fieldnames = ['site', 'author', 'title', 'text']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for filename in tqdm(os.listdir("../data/raw/webhose_political")):
        if filename.endswith(".json"):
            with open("../data/raw/webhose_political/" + filename, mode='r') as in_file:
                data = json.load(in_file)
                writer.writerow({'site': data['thread']['site'], 'author': data['author'], 'title': data['title'], 'text': data['text']})
