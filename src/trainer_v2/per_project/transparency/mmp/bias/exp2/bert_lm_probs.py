import torch
from transformers import BertForMaskedLM
from transformers import BertTokenizer, BertModel
from torch.nn import Softmax

from tab_print import print_table

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

text = "I am looking for a new car. Is [MASK]"
mask_idx = tokenizer.tokenize(text).index("[MASK]") + 1
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
# mask_idx = encoded_input.input_ids[0].find(103)
print(mask_idx)
print(encoded_input.input_ids[0, mask_idx] == 103)

car_maker_list = ['ford', 'buick', 'cadillac', 'porsche', 'jeep', 'fiat', 'bmw', 'chrysler', 'chevrolet', 'ferrari', 'volvo', 'toyota', 'hyundai', 'volkswagen', 'oldsmobile', 'mercedes', 'jaguar', 'mazda', 'renault', 'audi', 'packard', 'peugeot', 'honda', 'bentley', 'nissan', 'lexus', 'pontiac', 'daimler', 'mitsubishi']
probs = Softmax(dim=0)(output.logits[0, mask_idx])
car_maker_ids = tokenizer.convert_tokens_to_ids(car_maker_list)
res = [(term, float(probs[car_maker_ids[i]])) for i, term in enumerate(car_maker_list)]
print_table(res)



