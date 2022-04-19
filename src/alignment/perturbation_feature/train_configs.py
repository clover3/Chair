def get_pert_train_data_shape():
    seq_len = 256
    num_perturb = 9
    num_classes = 3
    shape = [seq_len, seq_len, num_perturb, num_classes]
    return shape

def get_pert_train_data_shape_1d():
    seq_len = 256
    num_perturb = 9
    num_classes = 3
    shape = [seq_len, num_perturb, num_classes]
    return shape