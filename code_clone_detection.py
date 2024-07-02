import os
import json


class CodeCloneDetection:
    
    def __init__(self, data_file, inference_class, model='phi3',
                 nl_instruction = 'do code 1 and code 2 solve identical problems with the same inputs and outputs ? answer with yes or no and no explanation.') -> None:
        self.model = model
        self.inference_class = inference_class
        self.data_file = data_file
        self.output_file = None
        self.data = self._read_data(data_file)
        self.gpt = self.inference_class(model=self.model, stream=False)
        self.prompts = [self.gpt.make_prompt(d['id'], d['code1'], d['code2'], nl_instruction) for d in self.data]
        
    
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
        code1:
        {code1}
        code2:
        {code2}
        {nl_instruction}
        """
        return (id, prompt)
    
    def run_processing(self, requested_samples_file, output_file):
        self.output_file = output_file 
        self.gpt.process_prompts(self.prompts, requested_samples_file, output_file)
        return self
