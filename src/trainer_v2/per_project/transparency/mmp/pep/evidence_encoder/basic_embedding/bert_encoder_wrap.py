import torch
from transformers import BertTokenizer, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from trainer_v2.chair_logging import c_log


class BERTEncoderWrap:
    def __init__(self, return_key="pooler_output"):
        # Load pre-trained model tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.return_key = return_key

    # Function to get the embedding of a word
    def batch_encode(self, words):
        c_log.info("Encoding for %d words", len(words))
        encoded_input = self.tokenizer(words, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        with torch.no_grad():
            outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.model(input_ids)
        if self.return_key == "first_token":
            ret = outputs.last_hidden_state[:, 0].cpu().numpy()
        elif self.return_key == "max":
            max_pool, _ = torch.max(outputs.last_hidden_state, dim=1)
            ret = max_pool.cpu().numpy()
        else:
            ret = outputs[self.return_key].cpu().numpy()
        c_log.info("Done")
        return ret