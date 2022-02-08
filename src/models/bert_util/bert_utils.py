import os


def batch2feed_dict_4_or_5_inputs(model, batch):
    if len(batch) == 4:
        x0, x1, x2, x3 = batch
        feed_dict = {
            model.x_list[0]: x0,
            model.x_list[1]: x1,
            model.x_list[2]: x2,
            model.x_list[3]: x3,
        }
    else:
        x0, x1, x2, x3, y = batch
        feed_dict = {
            model.x_list[0]: x0,
            model.x_list[1]: x1,
            model.x_list[2]: x2,
            model.x_list[3]: x3,
            model.y: y,
        }
    return feed_dict


def get_last_id(save_dir):
    print("searching: ", save_dir)
    last_model_id = None
    for (dirpath, dirnames, filenames) in os.walk(save_dir):
        for filename in filenames:
            if ".meta" in filename:
                print(filename)
                model_id = filename[:-5]
                if last_model_id is None:
                    last_model_id = model_id
                else:
                    last_model_id = model_id if model_id > last_model_id else last_model_id
    return last_model_id