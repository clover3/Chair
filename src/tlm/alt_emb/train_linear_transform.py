import tensorflow as tf

from tlm.alt_emb.show_embedding_difference import get_nli_and_bert_embeddings


def make_embeddings_as_dataset():
    bert_emb, nli_emb = get_nli_and_bert_embeddings()
    _, hidden_size = bert_emb.shape

    def pair_gen():
         for i in range(len(bert_emb)):
             yield bert_emb[i], nli_emb[i]

    def x_gen():
         for i in range(len(bert_emb)):
             yield bert_emb[i]

    def y_gen():
         for i in range(len(bert_emb)):
             yield nli_emb[i]

    x_series = tf.data.Dataset.from_generator(
        x_gen,
        output_types=tf.float32,
        output_shapes=(hidden_size,)
    )
    y_series = tf.data.Dataset.from_generator(
        y_gen,
        output_types=tf.float32,
        output_shapes=(hidden_size,)
    )

    return tf.data.Dataset.zip((x_series, y_series))


def build_model(decay_steps):
    num_voca = 30522
    hidden_size = 768

    model = tf.keras.Sequential([
        #tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=[hidden_size]),
        tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=[hidden_size]),
        tf.keras.layers.Dense(hidden_size),
    ])
    global_step = tf.compat.v1.train.get_or_create_global_step()
    learning_rate = tf.compat.v1.train.exponential_decay(1e-4,
                                                        global_step,
                                                        decay_steps, 1e-5,
                                                        )

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model.compile(loss='mae',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model



def train():
    tf.debugging.set_log_device_placement(True)

    dataset = make_embeddings_as_dataset()

    dataset.shuffle(1000)
    train_size = 25000
    train_dataset = dataset.take(train_size)

    num_epochs = 100
    train_batch_size = 300
    decay_steps = int(train_size/ train_batch_size) * num_epochs
    train_dataset = train_dataset.batch(train_batch_size)

    test_dataset = dataset.skip(train_size)
    test_dataset = test_dataset.batch(1000)
    model = build_model(decay_steps)
    model.evaluate(test_dataset)
    model.fit(train_dataset, epochs=num_epochs)
    # model.evaluate(test_dataset)


if __name__ == "__main__":
    train()