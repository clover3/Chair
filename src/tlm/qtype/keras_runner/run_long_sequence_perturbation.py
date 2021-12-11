import os

from cpath import output_path, data_path
from data_generator.light_dataloader import LightDataLoader
from tf_v2_support import disable_eager_execution
from tlm.qtype.keras_runner.qtype_model_functional import load_qde4
from trainer.np_modules import get_batches_ex


class ModelConfig:
    max_seq_length = 512
    q_type_voca = 2048


def prepare_model():
    model_config = ModelConfig()
    save_path = os.path.join(output_path, "model", "runs", "qtype_2T", "model.ckpt-200000")
    model = load_qde4(save_path, model_config)

    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(data_path, vocab_filename)
    data_loader = LightDataLoader(model_config.max_seq_length, voca_path)

    def predict(query, content_span, document):
        def get_inputs(sent1, sent2):
            data = list(data_loader.from_pairs([(sent1, sent2)]))
            batch = get_batches_ex(data, 1, 4)[0]
            x0, x1, x2, y = batch
            return (x0, x1, x2)

        qx0, qx1, qx2 = get_inputs(content_span, query)
        dx0, dx1, dx2 = get_inputs(content_span, document)
        inputs = qx0, qx1, qx2, dx0, dx1, dx2
        logits = model.predict(inputs)
        return logits

    return predict


def main():
    disable_eager_execution()
    predict = prepare_model()
    # predict = get_predictor()

    query = "what are the requirements to become a scribe"
    document = """scribe resources faq " q : what if i have a question that is not on this page ? q : what is a scribe ? q : do scribes provide direct patient care ? q : do medical scribes ompensation coverage ? q : do scribes receive vision and dental insurance coverage . q : do scribe receive retirement benefits ? q : can i turn becoming a scribe into a carmove around to different hospitals as needed ? q : is the scribe position a good fit for a college student ? is the position flexible ? q : can i work only during winter anocess for a new scribe ? q : are scribes compensated for training time ? q : how do i apply for a scribe position with scribe america ? q : what are some of the characteriscribe , is there opportunity for growth within scribe america ? q : what if i have a question that is not on this page ? a : all applicant questions go to applicant @ scribribe ? a : a scribe is a physician collaborator who fulfills the primary secretarial and non - clinical functions of the busy physician or mid - level provider . scribes spcity to provide direct patient care like seeing the next waiting patient , performing medical procedures and communicating with nursing staff . the scribe actively monitors. back to top q
Content_sapn: requirements to become a scribe"""
    content_span = "requirements to become a scribe"
        # query = input("Query: ")
        # document = input("Passage: ")
        # content_span = input("Content_span: ")
    logits = predict(query, content_span, document)
    print((logits))


if __name__ == "__main__":
    main()