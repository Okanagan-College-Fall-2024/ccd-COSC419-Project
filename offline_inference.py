import os
import json
import pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from analyse import Analyser
from model_inference import ModelInterface


current_location = pathlib.Path(__file__).parent.resolve()
quantization_config = BitsAndBytesConfig(load_in_4bit=True)


class OfflineRequest(ModelInterface):
    def __init__(self,  model="Phi-3-medium-128k-instruct", stream=False):
        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(current_location, model),
            quantization_config = quantization_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(current_location, model))
        self.stream = stream
        
        self.results = []

    def _send_request(self, prompt_id, prompt):
        torch.cuda.empty_cache()
        pad_token_id = self.tokenizer.eos_token_id
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", padding=True)
        inputs = inputs.to('cuda')
        input_ids_length = inputs.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                max_new_tokens=400, 
                temperature=0.7, 
                top_k=50,
                top_p=0.9, 
                no_repeat_ngram_size=4,
                do_sample=True )
            
        print(f"The request status for prompt id {prompt_id} is {self.tokenizer.decode(outputs[0, input_ids_length:], skip_special_tokens=False)}\n")
        return self.tokenizer.decode(outputs[0, input_ids_length:], skip_special_tokens=False)
    
    def make_prompt(self, id, code1, code2, nl_instruction):
        prompt = f"""
        <|user|> 
        code1:
        {code1}
        code2:
        {code2}
        {nl_instruction}
        <|end|>
        <|assistant|>
        """
        return (id, prompt)


