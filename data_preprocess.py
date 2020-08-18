# 資料前處理
# 把text 輸入變成 BERT 所需要的輸入
import csv
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset


def convert_data_to_feature(filename,taipe):
    data_list=[]
    label_list=[]
    i=0
    if taipe=='train':
        pos_value=500000
    else:
        pos_value=100000
    with open(filename,'r',newline="",encoding='utf-8') as csvfile:
        rows=csv.reader(csvfile)

        for row in rows:
            i+=1
            data_list.append(row[0])
            if i<pos_value:
                label_list.append(1)
            else:
                label_list.append(-1)
            

    assert len(data_list)==len(label_list)

    # 製作BERT input embeddings
    mode_version='bert-base-uncased'
    tokenizer=BertTokenizer.from_pretrained(mode_version)
    input_ids=[]
    max_seq_len=0 # 紀錄最大長度
    origin_lengh=[]# 紀錄原本長度
    for data in data_list:
        # 轉成token
        word_piece_list=tokenizer.tokenize(data)
        # 轉成ID
        input_id=tokenizer.convert_tokens_to_ids(word_piece_list)
        # 加上 CLS 跟 SEP
        input_id=tokenizer.build_inputs_with_special_tokens(input_id)

        if len(input_id)>max_seq_len:
            max_seq_len=len(input_id)
            
        input_ids.append(input_id)
    print("句子最大長度:",max_seq_len)
    assert max_seq_len <= 512

    # 補齊長度
    for input_id in input_ids:
        origin_lengh.append(len(input_id))
        while len(input_id)<max_seq_len:
            input_id.append(0) 

    # Segment 因為只有一句話所以id 都是 0 如果有第二句 則id=1
    segment_ids=[[0]*max_seq_len for i in range(len(data_list))]
    # position_id 又稱為 attention_id 代表要關注在哪裡
    position_ids=[]
    for i in range(len(data_list)):
        position_id=[]
        for j in range(origin_lengh[i]):
            position_id.append(1) # 1 代表要關注
        while (len(position_id)<max_seq_len):
            position_id.append(0) # 0 代表不須關注
        position_ids.append(position_id)

    assert len(input_ids) == len(segment_ids) and len(input_ids) == len(position_ids) and len(input_ids) == len(label_list)

    data_features={
        'input_ids':input_ids,
        'segment_ids':segment_ids,
        'position_ids':position_ids,
        'labels':label_list
    }

    return data_features

# 轉成 Pytorch 的輸入形式
def makeDataset(data_features):
    input_ids=data_features["input_ids"]
    segment_ids=data_features["segment_ids"]
    position_ids=data_features["position_ids"]
    labels=data_features["labels"]

    all_input_ids=torch.tensor([input_id for input_id in input_ids],dtype=torch.long)
    all_segment_ids=torch.tensor([segment_id for segment_id in segment_ids],dtype=torch.long)
    all_position_ids=torch.tensor([position_id for position_id in position_ids],dtype=torch.long)
    all_labels=torch.tensor([label for label in labels],dtype=torch.long)
    dataset=TensorDataset(all_input_ids,all_segment_ids,all_position_ids,all_labels)

    return dataset
if __name__ == "__main__":
    filename="/root/project/BERT_for_Sentiment/datasets/pos.csv"
    data_features=convert_data_to_feature(filename,1)
    dataset=makeDataset(data_features)