import csv


train_count=20000
test_count=2000
pos_path='/root/project/BERT_for_Sentiment/datasets/pos.csv'
neg_path='/root/project/BERT_for_Sentiment/datasets/neg.csv'

train_data=[]
test_data=[]
with open(pos_path,'r',newline="",encoding='utf-8') as posfile:
    rows=csv.reader(posfile)
    i=0
    k=0
    for row in rows:
        if i<train_count:
            train_data.append(row[0])
            i+=1
        elif k <test_count:
            test_data.append(row[0])
            k+=1
        if i==train_count and k ==test_count:
            break
with open(neg_path,'r',newline="",encoding='utf-8') as negfile:
    rows=csv.reader(negfile)
    i=0
    k=0
    for row in rows:
        if i<train_count:
            train_data.append(row[0])
            i+=1
        elif k <test_count:
            test_data.append(row[0])
            k+=1
        if i==train_count and k ==test_count:
            break

with open('train_data.csv', 'w',newline='',encoding='utf-8') as trainfile:
    writer = csv.writer(trainfile)
    for index,data in  enumerate(train_data) :
        writer.writerow([data])

with open('test_data.csv', 'w',newline='',encoding='utf-8') as testfile:
    writer = csv.writer(testfile)
    for index,data in  enumerate(test_data) :
        writer.writerow([data])