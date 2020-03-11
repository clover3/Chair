import sys

from google.cloud import storage

client = None


class ParseFail(Exception):
    pass

def parse_step(model_name):
    head = "model.ckpt-"
    if model_name.startswith(head):
        post = model_name[len(head):]
        step = int(post.split(".")[0])
        return step
    else:
        raise ParseFail()

# gs://clovertpu/(dir_path)
def delete_model_except_last(dir_path):
    global client
    if client is None:
        client = storage.Client()
    step_list = []
    blob_iter = client.list_blobs("clovertpu", prefix=dir_path)
    for blob in blob_iter:
        postfix = blob.name.split("/")[-1]
        try:
            step = parse_step(postfix)
            step_list.append(step)
        except ParseFail:
            pass

    max_step = max(step_list)
    delete_blob = []
    preserve_blob = []
    for blob in client.list_blobs("clovertpu", prefix=dir_path):
        postfix = blob.name.split("/")[-1]
        f_del = False
        try:
            step = parse_step(postfix)
            if step != max_step:
                f_del = True
            else:
                print("Preserve last ", postfix)
        except ParseFail:
            pass
        if f_del:
            delete_blob.append(blob)
        else:
            preserve_blob.append(blob)

    if len(preserve_blob) < 6:
        raise Exception()
    for blob in delete_blob:
        blob.delete()
    print("Deleted {} objects from {}".format(len(delete_blob), dir_path))


if __name__ == "__main__":
    expected_prefix = "training/model"
    dir_path = sys.argv[1]
    if dir_path.startswith(expected_prefix):
        delete_model_except_last(dir_path)
    else:
        print("Unexpected path : ", expected_prefix)
