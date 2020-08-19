# Use BERT for Sentiment judgement

## 第一步驟

先用data_split.py　把neg.csv 跟 pos.csv 轉成train.csv test.csv

## 第二步驟

執行 train.py 來訓練model

## 第三步驟

可使用predict.py 來測試自己的model 

# 若是想跳過訓練部分 

* 先執行init.sh 下載已訓練好的model
* 直接執行predict.py