from transformers import AutoTokenizer, AutoModelForMaskedLM, TFAutoModelForMaskedLM
import tensorflow as tf

def main():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = TFAutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

    encoded_input = tokenizer(["Some text", "other text"], return_tensors="tf")
    print(encoded_input)
    output = model(encoded_input)

    # input_ids = tf.expand_dims(encoded_input["input_ids"], axis=0)
    # attention_mask = tf.expand_dims(encoded_input["attention_mask"], axis=0)
    # output = model({'input_ids': input_ids, 'attention_mask': attention_mask})
    print(output)


if __name__ == "__main__":
    main()