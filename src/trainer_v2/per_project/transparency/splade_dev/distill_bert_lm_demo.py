from transformers import TFAutoModelForMaskedLM, AutoTokenizer
model_type = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_type)
model = TFAutoModelForMaskedLM.from_pretrained(model_type)
some_text = "The capital of France is [MASK] ."
output = model(tokenizer(some_text, return_tensor="tf"))
print(output)



