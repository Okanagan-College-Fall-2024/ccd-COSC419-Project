import os
import json
import pathlib
from abc import ABC, abstractmethod


import requests

from analyse import Analyser
from model_inference import ModelInterface


current_location = pathlib.Path(__file__).parent.resolve()


class OllamaInference(ModelInterface):
    def __init__(self,  model='phi3', stream=False):
        self.model = model
        self.stream = stream
        self.results = []

    def _send_request(self, prompt_id, prompt):
        url = "http://localhost:11434/api/generate"
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "prompt": f"{prompt}",
            "options": {
                "num_ctx": 4096
            },
            "stream": self.stream
        }
        response = requests.post(url, headers=headers, json=data)
        print(f"The request status for prompt id {prompt_id} is {response.status_code}\n")
        return response.json()['response']
    
    def make_prompt(self, id, code1, code2, nl_instruction):
        prompt = f"""
        code1:
        {code1}
        code2:
        {code2}
        {nl_instruction}
        """
        return (id, prompt)



# if __name__ == "__main__":
#     ollama_request = CodeCloneDetection(
#         data_file=os.path.join(current_location, 'ruby_java_test_clone3.jsonl'),
#         ).run_processing(
#         os.path.join(current_location, 'results', 'requested_ids_0.1.txt'), 
#         os.path.join(current_location, 'results', 'results_java_01.txt')
#     )
    
    # assert 1 == 1
    # analyser1 = Analyser(
    #     ollama_request.data_file,
    #     ollama_request.output_file
    # )
    # analyser1.compute_metrics('Metrics for phi3 Cross language ccd', save_to_file=True)
