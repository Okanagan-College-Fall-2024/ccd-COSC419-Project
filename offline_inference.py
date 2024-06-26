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

    
# nl_instruction = 'Do code 1 and code 2 solve identical problems with the same inputs and outputs? Your answer should include "yes" if the condition is satisfied, even if the codes are written in different programming languages, and it must include "no" if the condition is not satisfied.') -> None:                  
class CodeCloneDetection:
    def __init__(self, data_file, model="Phi-3-medium-128k-instruct",
                 nl_instruction = 'Do code 1 and code 2 solve identical problems with the same inputs and outputs? Just say yes or no and do not explain your answer ') -> None:
        self.model = model
        self.data_file = data_file
        
        self.output_file = None
        self.data = self._read_data(data_file)
        self.prompts = [self._make_probmpt(d['id'], d['code1'], d['code2'], nl_instruction) for d in self.data]
        self.gpt = OfflineRequest(model=self.model, stream=False)
    
    def _get_requested_ids(self, file_name):
        requested_ids = []
        if not os.path.exists(file_name):
            return []
        with open(file_name, 'r') as file:
            for line in file:
                requested_ids.append(int(line.strip()))
            
        return requested_ids

    def _read_data(self, data_file):
        with open(data_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                
        return data

    def _make_probmpt(self, id, code1, code2, nl_instruction):
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
    
    def run_processing(self, requested_samples_file, output_file):
        self.output_file = output_file
        self.gpt.process_prompts(self.prompts, requested_samples_file, output_file)
        return self


if __name__ == "__main__":

    offline_request = CodeCloneDetection(
        data_file=os.path.join(os.path.dirname(__file__), 'ruby_java_test_clone3.jsonl'),
    ).run_processing(
        os.path.join('resutls_mini_shehata_fixed_medium', 'requested_ids_0.1.txt'), 
        os.path.join('resutls_mini_shehata_fixed_medium', 'results_fixed_medium_01.txt')
    )
    print("Ready to evaluate")
    assert 1 == 1
    analyser1 = Analyser(
        offline_request.data_file,
        offline_request.output_file
    )
    analyser1.compute_metrics('Metrics for phi3 medium Cross language ccd', save_to_file=True)
