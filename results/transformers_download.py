from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/Phi-3-mini-4k-instruct"  # Change this to the model you need

# Download and save the model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained('./saved_model')

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('./saved_model')