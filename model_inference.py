import os
from abc import abstractmethod

class ModelInterface:
    
    def process_prompts(self, prompts, requested_samples_file, output_file):
        requested_ids = self._get_requested_ids(requested_samples_file)
        for prompt in prompts:
            sample_id = prompt[0]
            if sample_id in requested_ids:
                continue
            
            os.makedirs(os.path.dirname(requested_samples_file), exist_ok=True)
            with open(requested_samples_file, 'a') as file:
                file.write(f"{sample_id}\n")
                
            result = self._send_request(prompt[0], prompt[1])
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'a') as file:
                file.write(f"***Data Id {sample_id}: {result.strip()}+++\n \n")
                
    def _get_requested_ids(self, requested_sample_file):
        requested_ids = []
        if not os.path.exists(requested_sample_file):
            return []
        with open(requested_sample_file, 'r') as file:
            for line in file:
                requested_ids.append(int(line.strip()))
            
        return requested_ids
        
    
    @abstractmethod
    def _send_request(self, prompt_id, prompt):
        pass
    
    @abstractmethod
    def make_prompt(self, id, code1, code2, nl_instruction):
        pass
