import collections
import re

import tensorflow as tf

from tf_util.tf_logging import tf_logging
from tlm.model.dual_model_common import triple_model_prefix2, dual_model_prefix1, triple_model_prefix3, \
    dual_model_prefix2


def get_bert_assignment_map(tvars, lm_checkpoint):
    lm_assignment_candidate = {}
    real_name_map = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
        targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
        lm_assignment_candidate[targ_name] = var
        tf_logging.debug("Init from lm_checkpoint : %s" % name)
        real_name_map[targ_name] = name

    assignment_map = {}
    initialized_variable_names = {}
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in lm_assignment_candidate:
                continue
            assignment_map[name] = lm_assignment_candidate[name]

            tvar_name = real_name_map[name]

            initialized_variable_names[tvar_name] = 1
            initialized_variable_names[tvar_name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def get_assignment_map_remap_from_v1(tvars, remap_prefix, lm_checkpoint):
    tf_logging.debug("get_assignment_map_remap_from_v1")
    assignment_candidate = {}
    real_name_map = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        if remap_prefix == top_scope:
            inner_name = "/".join(tokens[1:])
            targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", inner_name)
            targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
            assignment_candidate[targ_name] = var
            tf_logging.info("Init from v1 : %s" % name)
            real_name_map[targ_name] = name

    assignment_map = {}
    initialized_variable_names = {}
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in assignment_candidate:
                continue
            assignment_map[name] = assignment_candidate[name]
            tvar_name = real_name_map[name]
            initialized_variable_names[tvar_name] = 1
            initialized_variable_names[tvar_name + ":0"] = 1

    return assignment_map, initialized_variable_names


# checkpoint is from tf2.0
def get_assignment_map_remap_from_v2(tvars, remap_prefix, lm_checkpoint):
    tf_logging.debug("get_assignment_map_remap_from_v2")
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}
    real_name_map = {}

    assignment_candidate = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        if remap_prefix == top_scope:
            inner_name = "/".join(tokens[1:])
            targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", inner_name)
            targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
            assignment_candidate[targ_name] = var
            tf_logging.info("Init from v2 : %s" % name)
            real_name_map[targ_name] = name

    assignment_map = collections.OrderedDict()
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            simple_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
            simple_name = re.sub("dense[_]?\d*", "dense", simple_name)
            tf_logging.debug("Vars in TT : %s" % name)
            tf_logging.debug("map to -> : %s" % simple_name)

            if simple_name not in assignment_candidate:
                continue
            assignment_map[name] = assignment_candidate[simple_name]
            tf_logging.debug("Matched variables : %s" % name)

            real_name = real_name_map[simple_name]
            initialized_variable_names[real_name] = 1
            initialized_variable_names[real_name + ":0"] = 1

    return assignment_map, initialized_variable_names


def sero_from_bert(upper_align_idx, tvars, lm_checkpoint):
    def get_target_name(var_name):
        targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", var_name)
        targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
        tokens = targ_name.split("/")
        if tokens[0] == "sero":
            tokens[0] = "bert"

        if len(tokens) > 2:
            if tokens[1] == "lower":
                tokens[1] = "encoder"
            elif tokens[1] == "upper":
                str_layer, str_no = tokens[2].split("_")
                str_no = str(int(str_no) + upper_align_idx)
                tokens[1] = "encoder"
                tokens[2] = str_layer + "_" + str_no
        targ_name = "/".join(tokens)
        return targ_name

    assignment_candidate = {}
    real_name_map = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        targ_name = get_target_name(name)
        assignment_candidate[targ_name] = var
        tf_logging.info("Init from v1 : %s" % name)
        real_name_map[targ_name] = name

    assignment_map = {}
    initialized_variable_names = {}
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            tf_logging.info("Checkpoint Var : %s" % name)
            if name not in assignment_candidate:
                continue
            assignment_map[name] = assignment_candidate[name]
            tvar_name = real_name_map[name]
            initialized_variable_names[tvar_name] = 1
            initialized_variable_names[tvar_name + ":0"] = 1

    return assignment_map, initialized_variable_names




def sero_from_v2(tvars, lm_checkpoint):
    tf_logging.debug("sero_from_v2")
    def get_target_name(var_name):
        targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", var_name)
        targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
        tokens = targ_name.split("/")
        if tokens[0] == "sero":
            tokens[0] = "bert"

        if len(tokens) > 2:
            if tokens[1] == "lower":
                tokens[1] = "encoder"
            elif tokens[1] == "upper":
                str_layer, str_no = tokens[2].split("_")
                str_no = str(int(str_no) + 6)
                tokens[1] = "encoder"
                tokens[2] = str_layer + "_" + str_no
        targ_name = "/".join(tokens)
        return targ_name

    assignment_candidate = {}
    real_name_map = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        targ_name = get_target_name(name)
        assignment_candidate[targ_name] = var
        tf_logging.info("Init from v2 : %s" % name)
        real_name_map[targ_name] = name

    assignment_map = {}
    initialized_variable_names = {}
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            simple_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
            simple_name = re.sub("dense[_]?\d*", "dense", simple_name)

            tf_logging.info("Checkpoint Var : %s" % name)
            if simple_name not in assignment_candidate:
                continue
            assignment_map[name] = assignment_candidate[simple_name]
            tvar_name = real_name_map[simple_name]
            initialized_variable_names[tvar_name] = 1
            initialized_variable_names[tvar_name + ":0"] = 1

    return assignment_map, initialized_variable_names



def get_bert_nli_assignment_map(tvars, lm_checkpoint):
    lm_assignment_candidate = {}
    real_name_map = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
        targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
        targ_name = targ_name.replace("cls_dense/kernel", "output_weights")
        targ_name = targ_name.replace("cls_dense/bias", "output_bias")
        lm_assignment_candidate[targ_name] = var
        tf_logging.info("Init from lm_checkpoint : %s" % name)
        real_name_map[targ_name] = name

    assignment_map = {}
    initialized_variable_names = {}
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in lm_assignment_candidate:
                continue
            assignment_map[name] = lm_assignment_candidate[name]

            tvar_name = real_name_map[name]

            initialized_variable_names[tvar_name] = 1
            initialized_variable_names[tvar_name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def get_cls_assignment(tvars, lm_checkpoint):
    lm_assignment_candidate = {}
    real_name_map = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
        targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
        lm_assignment_candidate[targ_name] = var
        real_name_map[targ_name] = name

    assignment_map = {}
    initialized_variable_names = {}
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in lm_assignment_candidate:
                continue
            if not name.startswith("cls"):
                continue
            assignment_map[name] = lm_assignment_candidate[name]

            tvar_name = real_name_map[name]

            initialized_variable_names[tvar_name] = 1
            initialized_variable_names[tvar_name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def get_tlm_assignment_map(tvars, tlm_prefix, lm_checkpoint, target_task_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}
    real_name_map = {}

    target_task_name_to_var = collections.OrderedDict()
    lm_assignment_candidate = {}
    tt_assignment_candidate = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        if tlm_prefix == top_scope:
            inner_name = "/".join(tokens[1:])
            target_task_name_to_var[inner_name] = var
            targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", inner_name)
            targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
            tt_assignment_candidate[targ_name] = var
            tf_logging.info("Init from target_task_checkpoint : %s" % name)
        else:
            targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
            targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
            lm_assignment_candidate[targ_name] = var
            tf_logging.info("Init from lm_checkpoint : %s" % name)

        real_name_map[targ_name] = name

    assignment_map_tt = collections.OrderedDict()
    if target_task_checkpoint:
        for x in tf.train.list_variables(target_task_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in tt_assignment_candidate:
                continue
            assignment_map_tt[name] = tt_assignment_candidate[name]

            real_name = real_name_map[name]
            initialized_variable_names[real_name] = 1

    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in lm_assignment_candidate:
                continue
            assignment_map[name] = lm_assignment_candidate[name]
            initialized_variable_names[name] = 1
            initialized_variable_names[name + ":0"] = 1

    return assignment_map, assignment_map_tt, initialized_variable_names

# target_task_checkpoint is from tf2.0
# I believe this function is equivalent to combination of following two functions
# - get_bert_assignment_map
# - get_assignment_map_remap_from_v2
def get_tlm_assignment_map_v2(tvars, tlm_prefix, lm_checkpoint, target_task_checkpoint_tf2):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}
    real_name_map = {}

    target_task_name_to_var = collections.OrderedDict()
    lm_assignment_candidate = {}
    tt_assignment_candidate = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        if tlm_prefix == top_scope:
            inner_name = "/".join(tokens[1:])
            target_task_name_to_var[inner_name] = var
            simple_name = inner_name
            simple_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", simple_name)
            simple_name = re.sub("dense[_]?\d*", "dense", simple_name)
            tt_assignment_candidate[simple_name] = var
            tf_logging.debug("Variable to be loaded from target_task_checkpoint : %s" % name)
            real_name_map[simple_name] = name
        else:
            simple_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
            simple_name = re.sub("dense[_]?\d*", "dense", simple_name)
            lm_assignment_candidate[simple_name] = var
            tf_logging.debug("Variable to be loaded from lm_checkpoint : %s" % name)
            real_name_map[simple_name] = name


    assignment_map_tt = collections.OrderedDict()
    if target_task_checkpoint_tf2:
        for x in tf.train.list_variables(target_task_checkpoint_tf2):
            (name, var) = (x[0], x[1])
            simple_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
            simple_name = re.sub("dense[_]?\d*", "dense", simple_name)
            tf_logging.debug("Vars in TT : %s" % name)
            tf_logging.debug("map to -> : %s" % simple_name)

            if simple_name not in tt_assignment_candidate:
                continue
            assignment_map_tt[name] = tt_assignment_candidate[simple_name]

            real_name = real_name_map[simple_name]
            initialized_variable_names[real_name] = 1

    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in lm_assignment_candidate:
                continue
            assignment_map[name] = lm_assignment_candidate[name]
            real_name = real_name_map[name]
            initialized_variable_names[real_name] = 1
            initialized_variable_names[real_name + ":0"] = 1

    return assignment_map, assignment_map_tt, initialized_variable_names

# checkpoint is from tf2.0
def assignment_map_v2_to_v2(tvars, lm_checkpoint_v2):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}
    real_name_map = {}
    tf_logging.debug("assignment_map_v2_to_v2")

    lm_assignment_candidate = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        simple_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
        simple_name = re.sub("dense[_]?\d*", "dense", simple_name)
        lm_assignment_candidate[simple_name] = var
        tf_logging.debug("Variable to be loaded from lm_checkpoint : %s" % name)
        tf_logging.debug("                            simple_name  : %s" % simple_name)
        real_name_map[simple_name] = name

    assignment_map = collections.OrderedDict()
    if lm_checkpoint_v2:
        for x in tf.train.list_variables(lm_checkpoint_v2):
            (name, var) = (x[0], x[1])
            simple_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
            simple_name = re.sub("dense[_]?\d*", "dense", simple_name)
            tf_logging.debug("Vars in TT : %s" % name)
            tf_logging.debug("map to -> : %s" % simple_name)

            if simple_name not in lm_assignment_candidate:
                continue
            assignment_map[name] = lm_assignment_candidate[simple_name]
            tf_logging.debug("Matched variables : %s" % name)

            real_name = real_name_map[simple_name]
            initialized_variable_names[real_name] = 1
            initialized_variable_names[real_name + ":0"] = 1

    return assignment_map, initialized_variable_names


def assignment_map_v2_to_v2_only_attention(tvars, lm_checkpoint_v2):
    def allow(var_name):
        return "attention" in var_name or "embedding" in var_name

    tvars = [v for v in tvars if allow(v.name)]
    return assignment_map_v2_to_v2(tvars, lm_checkpoint_v2)


def get_assignment_map_as_is(tvars, checkpoint):
    current_vars = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        current_vars[name] = var
        tf_logging.debug("Init from lm_checkpoint : %s" % name)

    assignment_map = {}
    initialized_variable_names = {}
    if checkpoint:
        for x in tf.train.list_variables(checkpoint):
            (name, var) = (x[0], x[1])
            if name not in current_vars:
                continue
            assignment_map[name] = current_vars[name]
            tf_logging.debug("Mapped : %s" % name)

            initialized_variable_names[name] = 1
            initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names



def bert_assignment_only_attention(tvars, lm_checkpoint):
    lm_assignment_candidate = {}
    real_name_map = {}

    def allow(var_name):
        return "attention" in var_name or "embedding" in var_name

    for var in tvars:
        name = var.name
        if not allow(name):
            continue
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
        targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
        lm_assignment_candidate[targ_name] = var
        tf_logging.debug("Init from lm_checkpoint : %s" % name)
        real_name_map[targ_name] = name

    assignment_map = {}
    initialized_variable_names = {}
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in lm_assignment_candidate:
                continue
            assignment_map[name] = lm_assignment_candidate[name]

            tvar_name = real_name_map[name]

            initialized_variable_names[tvar_name] = 1
            initialized_variable_names[tvar_name + ":0"] = 1

    return (assignment_map, initialized_variable_names)



def bert_assignment_wo_attention(tvars, lm_checkpoint):
    lm_assignment_candidate = {}
    real_name_map = {}

    def allow(var_name):
        return "attention" not in var_name

    for var in tvars:
        name = var.name
        if not allow(name):
            continue
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
        targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
        lm_assignment_candidate[targ_name] = var
        tf_logging.info("Init from lm_checkpoint : %s" % name)
        real_name_map[targ_name] = name

    assignment_map = {}
    initialized_variable_names = {}
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in lm_assignment_candidate:
                continue
            assignment_map[name] = lm_assignment_candidate[name]

            tvar_name = real_name_map[name]

            initialized_variable_names[tvar_name] = 1
            initialized_variable_names[tvar_name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def get_init_fn_for_two_checkpoints(train_config, tvars, init_checkpoint, remap_prefix, second_init_checkpoint, remap_prefix2):
    # if train_config.checkpoint_type == "v2":
    assignment_fn1 = get_assignment_map_remap_from_v2
    # else:
    #     assignment_fn1 = get_assignment_map_remap_from_v1
    assignment_fn2 = get_assignment_map_remap_from_v2

    assignment_map, initialized_variable_names \
        = assignment_fn1(tvars, remap_prefix, init_checkpoint)

    assignment_map2, initialized_variable_names2 \
        = assignment_fn2(tvars, remap_prefix2, second_init_checkpoint)
    for k, v in initialized_variable_names2.items():
        initialized_variable_names[k] = v

    def init_fn():
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.compat.v1.train.init_from_checkpoint(second_init_checkpoint, assignment_map2)

    return initialized_variable_names, init_fn


def get_init_fn_for_two_checkpoints_ex(first_from_v1, second_from_v1,
                                          tvars, init_checkpoint,
                                          remap_prefix, second_init_checkpoint, remap_prefix2):
    assignment_fn1 = get_assignment_map_remap_from_v1 if first_from_v1 else get_assignment_map_remap_from_v2
    assignment_fn2 = get_assignment_map_remap_from_v1 if second_from_v1 else get_assignment_map_remap_from_v2

    assignment_map, initialized_variable_names \
        = assignment_fn1(tvars, remap_prefix, init_checkpoint)

    assignment_map2, initialized_variable_names2 \
        = assignment_fn2(tvars, remap_prefix2, second_init_checkpoint)
    for k, v in initialized_variable_names2.items():
        initialized_variable_names[k] = v

    def init_fn():
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.compat.v1.train.init_from_checkpoint(second_init_checkpoint, assignment_map2)

    return initialized_variable_names, init_fn



def get_init_fn_for_three_checkpoints(train_config,
                                      tvars,
                                      init_checkpoint,
                                      remap_prefix,
                                      second_init_checkpoint,
                                      remap_prefix2,
                                      third_init_checkpoint,
                                      remap_prefix3
                                      ):
    # if train_config.checkpoint_type == "v2":
    assignment_fn1 = get_assignment_map_remap_from_v2
    # else:
    #     assignment_fn1 = get_assignment_map_remap_from_v1
    assignment_fn2 = get_assignment_map_remap_from_v2
    assignment_fn3 = get_assignment_map_remap_from_v2


    assignment_map, initialized_variable_names \
        = assignment_fn1(tvars, remap_prefix, init_checkpoint)

    assignment_map2, initialized_variable_names2 \
        = assignment_fn2(tvars, remap_prefix2, second_init_checkpoint)

    assignment_map3, initialized_variable_names3 \
        = assignment_fn3(tvars, remap_prefix3, third_init_checkpoint)

    for k, v in initialized_variable_names2.items():
        initialized_variable_names[k] = v
    for k, v in initialized_variable_names3.items():
        initialized_variable_names[k] = v

    def init_fn():
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.compat.v1.train.init_from_checkpoint(second_init_checkpoint, assignment_map2)
        tf.compat.v1.train.init_from_checkpoint(third_init_checkpoint, assignment_map3)

    return initialized_variable_names, init_fn


def cppnc_assignment_remap2(tvars, lm_checkpoint):
    tf_logging.debug("get_assignment_map_remap_from_v2")
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}
    real_name_map = {}

    assignment_candidate = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        if triple_model_prefix2 == top_scope:
            targ_name = get_name_key(dual_model_prefix1, tokens)
            assignment_candidate[targ_name] = var
            tf_logging.info("Init from v2 : %s" % name)
            real_name_map[targ_name] = name
        elif triple_model_prefix3 == top_scope:
            targ_name = get_name_key(dual_model_prefix2, tokens)
            assignment_candidate[targ_name] = var
            tf_logging.info("Init from v2 : %s" % name)
            real_name_map[targ_name] = name

    assignment_map = collections.OrderedDict()
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            simple_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
            simple_name = re.sub("dense[_]?\d*", "dense", simple_name)
            tf_logging.debug("Vars in TT : %s" % name)
            tf_logging.debug("map to -> : %s" % simple_name)

            if simple_name not in assignment_candidate:
                continue
            assignment_map[name] = assignment_candidate[simple_name]
            tf_logging.debug("Matched variables : %s" % name)

            real_name = real_name_map[simple_name]
            initialized_variable_names[real_name] = 1
            initialized_variable_names[real_name + ":0"] = 1

    return assignment_map, initialized_variable_names


def phase2_to_phase1_assignment_remap(tvars, src_checkpoint):
    tf_logging.debug("get_assignment_map_remap_from_v2")
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}
    assignment_candidate = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        assignment_candidate[name] = var

    def parse_shift_name(name, key, shift_idx):
        tokens = name.split("/")
        new_tokens = []
        for token in tokens:
            if token.startswith(key):
                if token == key:
                    idx = 0
                else:
                    idx_str = token[len(key) + 1:]
                    idx = int(idx_str)
                new_idx = idx + shift_idx

                assert new_idx >= 0
                if new_idx == 0:
                    new_token = key
                else:
                    new_token = "_".join([key, str(new_idx)])
            else:
                new_token = token

            new_tokens.append(new_token)
        return "/".join(new_tokens)

    assignment_map = collections.OrderedDict()
    if src_checkpoint:
        for x in tf.train.list_variables(src_checkpoint):
            (old_name, var) = (x[0], x[1])
            if old_name.startswith(dual_model_prefix2):
                new_name = old_name.replace(dual_model_prefix2, dual_model_prefix1)
                if "/dense" in new_name:
                    new_name = parse_shift_name(new_name, "dense", -37)
                if "/layer_normalization" in new_name:
                    new_name = parse_shift_name(new_name, "layer_normalization", -25)
            elif old_name.startswith("cls_dense_1/"):
                new_name = old_name.replace("cls_dense_1/", dual_model_prefix1 + "/cls_dense/")
            else:
                new_name = None

            if new_name is not None and new_name in assignment_candidate:
                tf_logging.debug("Vars in checkpoint : %s" % old_name)
                tf_logging.debug("map to -> : %s" % new_name)

                assignment_map[old_name] = assignment_candidate[new_name]
                initialized_variable_names[new_name] = 1
                initialized_variable_names[new_name + ":0"] = 1

    return assignment_map, initialized_variable_names


def phase1_only_load(tvars, src_checkpoint):
    tf_logging.debug("phase1_only_load")
    initialized_variable_names = {}
    assignment_candidate = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        assignment_candidate[name] = var

    assignment_map = collections.OrderedDict()
    if src_checkpoint:
        for x in tf.train.list_variables(src_checkpoint):
            (name, var) = (x[0], x[1])

            if name.startswith(dual_model_prefix1):
                include = True
            elif name.startswith("cls_dense/"):
                include = True
            else:
                include = False

            if include and name in assignment_candidate:
                tf_logging.debug("Vars in checkpoint : %s" % name)
                tf_logging.debug("map to -> : %s" % name)
                assignment_map[name] = assignment_candidate[name]
                initialized_variable_names[name] = 1
                initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names


def get_name_key(head_scope, tokens):
    inner_name = "/".join([head_scope] + tokens[1:])
    targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", inner_name)
    targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
    return targ_name


def get_init_fn_for_cppnc_start(train_config, tvars, init_checkpoint, remap_prefix, second_init_checkpoint, remap_prefix2):
    assignment_fn1 = get_assignment_map_remap_from_v2

    assignment_map, initialized_variable_names \
        = assignment_fn1(tvars, remap_prefix, init_checkpoint)

    assignment_map2, initialized_variable_names2 \
        = cppnc_assignment_remap2(tvars, second_init_checkpoint)
    for k, v in initialized_variable_names2.items():
        initialized_variable_names[k] = v

    def init_fn():
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.compat.v1.train.init_from_checkpoint(second_init_checkpoint, assignment_map2)

    return initialized_variable_names, init_fn


def get_init_fn_for_phase2_phase1_remap(train_config, tvars, init_checkpoint, remap_prefix, second_init_checkpoint, remap_prefix2):
    assignment_fn2 = get_assignment_map_remap_from_v1

    assignment_map, initialized_variable_names \
        = phase2_to_phase1_assignment_remap(tvars, init_checkpoint)

    assignment_map2, initialized_variable_names2 \
        = assignment_fn2(tvars, remap_prefix2, second_init_checkpoint)

    for k, v in initialized_variable_names2.items():
        initialized_variable_names[k] = v

    def init_fn():
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.compat.v1.train.init_from_checkpoint(second_init_checkpoint, assignment_map2)

    return initialized_variable_names, init_fn


def get_init_fn_for_phase1_load_and_bert(train_config, tvars, init_checkpoint, remap_prefix, second_init_checkpoint, remap_prefix2):
    assignment_fn2 = get_assignment_map_remap_from_v1

    assignment_map, initialized_variable_names \
        = phase1_only_load(tvars, init_checkpoint)

    assignment_map2, initialized_variable_names2 \
        = assignment_fn2(tvars, remap_prefix2, second_init_checkpoint)

    for k, v in initialized_variable_names2.items():
        initialized_variable_names[k] = v

    def init_fn():
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.compat.v1.train.init_from_checkpoint(second_init_checkpoint, assignment_map2)

    return initialized_variable_names, init_fn

