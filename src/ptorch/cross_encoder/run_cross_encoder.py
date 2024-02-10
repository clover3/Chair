from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


q_list = ['How many people live in Berlin?', 'How many people live in Berlin?']
d_list=  [
    'Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.',
    'New York City is famous for the Metropolitan Museum of Art.']

features = tokenizer(q_list, d_list, padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    output = model(**features, output_attentions=True)
    scores = output.logits
    print(scores)
    print(features.input_ids)

    print(len(output.attentions))
    for t in output.attentions:
        print(t.shape)