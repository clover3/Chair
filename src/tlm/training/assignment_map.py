import collections
import re

import tensorflow as tf

from tf_util.tf_logging import tf_logging


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


def get_assignment_map_remap_from_v1(tvars, remap_prefix, lm_checkpoint):
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

