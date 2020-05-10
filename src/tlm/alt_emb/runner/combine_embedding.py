import os

from cpath import pjoin, output_path
from tlm.alt_emb.combine_embedding import combine


def combine_nli_alt_emb():
    model_dir = pjoin(output_path, "models")
    nli_checkpoint = pjoin(pjoin(model_dir, "nli_bert_300_K"), "model.ckpt-73150")
    #alt_emb_checkpoint = pjoin(pjoin(model_dir, "alt_emb_F"), "model.ckpt-10000")
    #alt_emb_checkpoint = pjoin(pjoin(model_dir, "alt_emb_F"), "model.ckpt-10000")
    #alt_emb_checkpoint = pjoin(pjoin(model_dir, "alt_emb_G"), "model.ckpt-0")
    alt_emb_checkpoint = pjoin(pjoin(model_dir, "alt_emb_G"), "model.ckpt-100000")

    save_path = os.path.join(output_path, "models", "nli_alt_emb_100KF", "model.ckpt-73150")
    combine(nli_checkpoint, alt_emb_checkpoint, save_path)


def alt_from_clueweb12_13A():
    model_dir = pjoin(output_path, "models")
    nli_checkpoint = pjoin(pjoin(model_dir, "nli_bert_300_K"), "model.ckpt-73150")
    alt_emb_checkpoint = pjoin(pjoin(model_dir, "alt_emb_H"), "model.ckpt-20000")
    save_path = os.path.join(output_path, "models", "nli_alt_emb_H20K", "model.ckpt-73150")
    combine(nli_checkpoint, alt_emb_checkpoint, save_path)


if __name__ == "__main__":
    alt_from_clueweb12_13A_100K()