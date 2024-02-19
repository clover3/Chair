from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
