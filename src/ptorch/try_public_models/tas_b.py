import torch


from transformers import AutoTokenizer, AutoModel

from ptorch.try_public_models.contriever import get_scorer_from_dual_encoder


def demo():
    # you can switch the model to the original "distilbert-base-uncased" to see that the usage example then breaks and the score ordering is reversed :O
    #pre_trained_model_name = "distilbert-base-uncased"
    pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
    bert_model = AutoModel.from_pretrained(pre_trained_model_name)

    # our relevant example

    passage1_input = tokenizer("We are very happy to show you the 🤗 Transformers library for pre-trained language models. We are helping the community work together towards the goal of advancing NLP 🔥.",return_tensors="pt")
    # a non-relevant example

    passage2_input = tokenizer("Hmm I don't like this new movie about transformers that i got from my local library. Those transformers are robots?",return_tensors="pt")
    # the user query -> which should give us a better score for the first passage

    query_input = tokenizer("what is the transformers library",return_tensors="pt")

    print("Passage 1 Tokenized:",tokenizer.convert_ids_to_tokens(passage1_input["input_ids"][0]))
    print("Passage 2 Tokenized:",tokenizer.convert_ids_to_tokens(passage2_input["input_ids"][0]))
    print("Query Tokenized:",tokenizer.convert_ids_to_tokens(query_input["input_ids"][0]))

    # note how we call the bert model independently between passages and query :)
    # [0][:,0,:] pools (or selects) the CLS vector from the full output

    bert_out = bert_model(**passage1_input)
    print(bert_out)
    print("bert_out[0].shape", bert_out[0].shape)
    passage1_encoded = bert_out[0][:,0,:].squeeze(0)
    passage2_encoded = bert_model(**passage2_input)[0][:,0,:].squeeze(0)
    query_encoded = bert_model(**query_input)[0][:,0,:].squeeze(0)
    print("---")
    print("Passage Encoded Shape:",passage1_encoded.shape)
    print("Query Encoded Shape:",query_encoded.shape)


def get_tas_b_encoder():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    def encode(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            inputs.to(device)
            outputs = model(**inputs)
            rep = outputs[0][:, 0, :]  # CLS Pooling
            return rep

    return encode


def get_tas_b_as_reranker():
    encoder = get_tas_b_encoder()
    score_fn = get_scorer_from_dual_encoder(encoder)
    return score_fn



def main():
    score_fn = get_tas_b_as_reranker()
    queries = [
        "Where was Marie Curie born?",
        "Where was John Foley born?",
    ]
    sentences = [
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
    ]
    qd = list(zip(queries, sentences))
    print(score_fn(qd))



    # demo()


if __name__ == "__main__":
    main()