import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Load the T5 model and tokenizer
model_name = "t5-small"  # You can choose other versions like 't5-base', 't5-large', etc.
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)

# Prepare the input text
input_text = "The term item is similar to "
input_ids = tokenizer.encode(input_text, return_tensors="tf")

num_candidates = 5

# Generate top-k candidate outputs
output = model.generate(input_ids,
                        max_length=50,
                        num_beams=num_candidates,
                        num_return_sequences=num_candidates,
                        early_stopping=True)

# Decode and print each candidate
for i, token in enumerate(output):
    decoded_output = tokenizer.decode(token, skip_special_tokens=True)
    print(f"Candidate {i+1}: {decoded_output}")
