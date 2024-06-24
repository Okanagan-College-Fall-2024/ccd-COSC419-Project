
# Knowledge Distillation For Large Language Models

## Experiments using Phi3-mini 
For this experiment, we have employed the ollama phi3-mini model. The data that we have used is the cross-language code clone detection data from our previous  [research using GPT-3.5](https://github.com/mkhfring/llm-for-ccd/blob/main/llm_for_ccd/ruby_java_test_clone3.jsonl). 

**results for phi3-Mini using Ollama**
  
| F1 score | Precision | Recall |
|:----------:|:----------:|:----------:|
| 0.51 | 0.70 | 0.40 |


**results for phi3-Mini using HuggingFace**
  
| F1 score | Precision | Recall |
|:----------:|:----------:|:----------:|
| 0.47 | 0.54 | 0.42 |


**results for phi3-Medium using HuggingFaxce**
  
| F1 score | Precision | Recall |
|:----------:|:----------:|:----------:|
| 0.72 | 0.66 | 0.80 |
