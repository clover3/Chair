

from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf

def main_case():
    model_name = "distilbert-base-uncased"
    sequence_classification_model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    input_ids = tf.zeros([2, 4], dtype=tf.int32)
    with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
        cls_output = sequence_classification_model(input_ids, output_attentions=True, output_hidden_states=True)
        tape.watch(cls_output.attentions)

    g = tape.gradient(cls_output.logits, cls_output.attentions)
    print(g)



def other_case():
    x = tf.constant([[3.0, 0.3]])

    with tf.GradientTape() as tape:
        input_ids = tf.keras.layers.Input(shape=(4,), dtype=tf.float32, name="input_ids")
        dense_layer = tf.keras.layers.Dense(1)
        tape.watch(dense_layer.trainable_variables)
        output = dense_layer(x)
    g = tape.gradient(output, dense_layer.trainable_variables)
    print(g)


def main():
    main_case()


if __name__ == "__main__":
    main()
