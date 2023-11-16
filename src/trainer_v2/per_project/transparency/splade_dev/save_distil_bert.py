import tensorflow as tf
from transformers import TFAutoModelForMaskedLM


from cpath import get_canonical_model_path

def main():
    "distilbert-base-uncased"
    model = TFAutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
    model.save(get_canonical_model_path("distilbert-base-uncased"))


def do_load():
    "distilbert-base-uncased"
    model = tf.keras.models.load_model(get_canonical_model_path("distilbert-base-uncased"))
    print(model)



if __name__ == "__main__":
    main()