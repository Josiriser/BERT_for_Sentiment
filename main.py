import csv
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

PRETRAINED_MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# 開啟 CSV 檔案
with open('/root/project/datasets/pos.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        text = row[0]
        break
CLS="[CLS]"
text=CLS+text
print(text)
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(tokens)
print(ids)
