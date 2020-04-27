import os

from cpath import output_path
from misc_lib import exist_or_mkdir
from tlm.alt_emb.gs_combine_embedding import download_and_combine, upload_to_gs


def ehealth_K():
    model_1_path = "gs://clovertpu/training/model/ehealth_bert_freeze/model.ckpt-10000"
    model_2_path = 'gs://clover_eu4/model/alt_emb_K/model.ckpt-20000'
    save_dir = os.path.join(output_path, "ehealth_K")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, "model.ckpt-0")
    download_and_combine(model_1_path, model_2_path, save_path)
    upload_gs_dir = "gs://clover_eu4/model/ehealth_combined_K"
    upload_to_gs(save_path, upload_gs_dir)


def ehealth_5_K():
    model_1_path = "gs://clovertpu/training/model/ehealth_bert_freeze5/model.ckpt-10000"
    model_2_path = 'gs://clover_eu4/model/alt_emb_K/model.ckpt-20000'
    save_dir = os.path.join(output_path, "ehealth_5_K")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, "model.ckpt-0")
    download_and_combine(model_1_path, model_2_path, save_path)
    upload_gs_dir = "gs://clover_eu4/model/ehealth_5_K"
    upload_to_gs(save_path, upload_gs_dir)



def do_nli():
    model_1_path = "gs://clovertpu/training/model/nli_bert_freeze_D/model.ckpt-73615"
    model_2_path = 'gs://clover_eu4/model/alt_emb_L/model.ckpt-20000'
    save_dir = os.path.join(output_path, "nli_from_alt_emb_L")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, "model.ckpt-0")
    download_and_combine(model_1_path, model_2_path, save_path)
    upload_gs_dir = "gs://clover_eu4/model/nli_from_alt_emb_L"
    upload_to_gs(save_path, upload_gs_dir)


if __name__ == "__main__":
    ehealth_5_K()

