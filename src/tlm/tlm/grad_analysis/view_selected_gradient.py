import pickle
import sys

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
from misc_lib import get_dir_files
from tlm.token_utils import mask_resolve_1, cells_from_tokens, is_mask
from visualize.html_visual import HtmlVisualizer


def loss_view(dir_path):
    tokenizer = get_tokenizer()
    html_writer = HtmlVisualizer("ukp_lm_grad_high.html", dark_mode=False)

    for file_path in get_dir_files(dir_path):
        items = pickle.load(open(file_path, "rb"))

        for e in items:
            input_ids, masked_input_ids, masked_lm_example_loss = e
            tokens = mask_resolve_1(tokenizer.convert_ids_to_tokens(input_ids),
                                    tokenizer.convert_ids_to_tokens(masked_input_ids))
            highlight = lmap(is_mask, tokens)

            cells = cells_from_tokens(tokens, highlight)
            html_writer.multirow_print(cells)


if __name__ == '__main__':
    loss_view(sys.argv[1])