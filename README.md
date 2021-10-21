# ContextRewriter

**This code is for AAAI 2021 paper [Contextualized Rewriting for Text Summarization](https://ojs.aaai.org/index.php/AAAI/article/view/17487)**

**Python Version**: Python3.6

**Package Requirements**: torch==1.1.0 pytorch_transformers tensorboardX multiprocess pyrouge

Some codes are borrowed from [ONMT](https://github.com/OpenNMT/OpenNMT-py) and [PreSumm](https://github.com/nlpyang/PreSumm).


## Results
Contextualized rewriter applied to various extractive summarizers on CNN/DailyMail (30/9/2020):
    
<table class="tg"  style="padding-left: 30px;">
  <tr>
    <th class="tg-0pky">Models</th>
    <th class="tg-0pky">ROUGE-1</th>
    <th class="tg-0pky">ROUGE-2</th>
    <th class="tg-0pky">ROUGE-L</th>
    <th class="tg-0pky">Words</th>
  </tr>
  <tr>
    <td class="tg-0pky">Oracle of BERT-Ext</td>
    <td class="tg-0pky">46.77</td>
    <td class="tg-0pky">26.78</td>
    <td class="tg-0pky">43.32</td>
    <td class="tg-0pky">112</td>
  </tr>
  <tr>
    <td class="tg-0pky"> + ContextRewriter</td>
    <td class="tg-0pky">52.57 (+5.80)</td>
    <td class="tg-0pky">29.71 (+2.93)</td>
    <td class="tg-0pky">49.69 (+6.37)</td>
    <td class="tg-0pky">63</td>
  </tr>
  <tr>
    <td class="tg-0pky">LEAD-3</td>
    <td class="tg-0pky">40.34</td>
    <td class="tg-0pky">17.70</td>
    <td class="tg-0pky">36.57</td>
    <td class="tg-0pky">85</td>
  </tr>
  <tr>
    <td class="tg-0pky"> + ContextRewriter</td>
    <td class="tg-0pky">41.09 (+0.75)</td>
    <td class="tg-0pky">18.19 (+0.49)</td>
    <td class="tg-0pky">38.06 (+1.49)</td>
    <td class="tg-0pky">55</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERTSUMEXT w/o Tri-Bloc</td>
    <td class="tg-0pky">42.50</td>
    <td class="tg-0pky">19.88</td>
    <td class="tg-0pky">38.91</td>
    <td class="tg-0pky">80</td>
  </tr>
  <tr>
    <td class="tg-0pky"> + ContextRewriter</td>
    <td class="tg-0pky">43.31 (+0.81)</td>
    <td class="tg-0pky">20.44 (+0.56)</td>
    <td class="tg-0pky">40.33 (+1.42)</td>
    <td class="tg-0pky">54</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT-Ext (ours)</td>
    <td class="tg-0pky">41.04</td>
    <td class="tg-0pky">19.56</td>
    <td class="tg-0pky">37.66</td>
    <td class="tg-0pky">105</td>
  </tr>
  <tr>
    <td class="tg-0pky"> + ContextRewriter</td>
    <td class="tg-0pky">43.52 (+2.48)</td>
    <td class="tg-0pky">20.57 (+1.01)</td>
    <td class="tg-0pky">40.56 (+2.90)</td>
    <td class="tg-0pky">66</td>
  </tr>
</table>

## Model Evaluation
Contextualized rewriter can be evaluated through this experimental scripts. 
The Lead3, BERTSUMEXT, and BERT-Ext extractive summarizers are included. 
All the parameters and settings are hard-coded in the py file.
```
    python src/exp_varext_guidabs.py 
```

The rewriter can also be easily applied to other extractive summarizer using following code.
The full example can be found in context_rewriter.py.
```
    rewriter = ContextRewriter(args.model_file)
    
    doc_lines = ["georgia high school ...", "less than 24 hours ...", ...]
    ext_lines = ["georgia high school ...", "less than 24 hours ..."]
    res_lines = rewriter.rewrite(doc_lines, ext_lines)
```
    
## Model Training
Contextualized rewriter can be trained with following script. 
All the settings are packed into the .py file.
```
    python src/exp_guidabs.py
```
By default, the input data path is ./bert_data, and the output model path is ./exp_guidabs