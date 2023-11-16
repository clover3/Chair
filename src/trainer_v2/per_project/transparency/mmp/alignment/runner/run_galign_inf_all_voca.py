import os


from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.mmp.alignment.galign_inf_helper import build_dataset_q_term_d_term

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from trainer_v2.chair_logging import c_log
import tensorflow as tf
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config_for_predict
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config_for_predict(args)
    run_config.print_info()
    tokenizer = get_tokenizer()

    strategy = get_strategy_from_config(run_config)
    d_term_id_st = 1000
    d_term_id_ed = 30000

    with strategy.scope():
        model = tf.keras.models.load_model(run_config.predict_config.model_save_path, compile=False)

    while True:
        q_term = input("Enter query term: ")
        q_term_id = tokenizer.convert_tokens_to_ids([q_term])[0]
        eval_dataset = build_dataset_q_term_d_term(q_term_id, d_term_id_st, d_term_id_ed)
        batched_dataset = eval_dataset.batch(run_config.common_run_config.batch_size)
        outputs = model.predict(batched_dataset)
        scores = outputs['align_probe']['all_concat']
        preds = tf.less(0, scores)
        c_log.info("q_term %s", q_term,)
        preds = preds.numpy()
        pos = []
        for i, record in enumerate(eval_dataset):
            pred = preds[i]
            if pred:
                d_term = record['d_term'][0].numpy().tolist()
                pos.append(d_term)
        # for i in preds.nonzero():
        #     pos.append(i[0])
        c_log.info("Done")
        print(tokenizer.convert_ids_to_tokens(pos))


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)