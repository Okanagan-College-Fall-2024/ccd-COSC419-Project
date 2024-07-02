import os
import argparse

from ollama_inference import OllamaInference
from offline_inference import OfflineRequest
from code_clone_detection import CodeCloneDetection


current_models = {
    'medium': 'Phi-3-medium-128k-instruct',
    'mini': 'Phi-3-mini-128k-instruct'
}


def main():
    parser = argparse.ArgumentParser(description='Run Code Clone Detection')
    parser.add_argument('--model_class', type=str, required=True, choices=['offline', 'ollama'], help='The model class to use')
    parser.add_argument('--data_file', type=str, default='ruby_java_test_clone3.jsonl', help='The data file containing code samples')
    parser.add_argument('--result_dir', type=str, default='model_result', help='The location to save the results')
    parser.add_argument('--model', type=str, default='phi3', help='The model to use')

    args = parser.parse_args()
    model = 'phi3'
    # Determine the model class
    if args.model_class == 'offline':
        model_cls = OfflineRequest
        model = current_models[args.model]
    elif args.model_class == 'ollama':
        model_cls = OllamaInference
    else:
        raise ValueError(f"Unknown model class: {args.model_class}")
    
    print(args.data_file, model)


    # Create an instance of CodeCloneDetection
    detection = CodeCloneDetection(data_file=args.data_file, inference_class=model_cls, model=model)

    # Run the processing
    detection.run_processing(
        requested_samples_file=os.path.join(args.result_dir, 'requested_ids.txt'), 
        output_file=os.path.join(args.result_dir, f'{args.model}_results.txt'), 
    )

if __name__ == "__main__":
    main()