# Data Preparation

Download the [pre-processed CNN/DM](https://drive.google.com/file/d/173_3qIV_A0pURh130dDfL-P1A4L_KFEE/view) and extract it into bert_data folder.

Or you can pre-process the data by yourself following below steps.

Steps: 
1) Following [PreSumm](https://github.com/nlpyang/PreSumm) for preparing initial data and put it into folder ./source_bert_data.
2) Run the command to convert ./source_bert_data to ./bert_data:
```
    python src/prepro/data_builder.py
``` 
