
import typing

import matchzoo
import pandas as pd

from arg.perspectives.classification_header import get_file_path


def load_data(
    stage: str = 'train',
    task: str = 'classification',
    target_label: str = 'dummy',
    return_classes: bool = False
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load SNLI data.

    :param stage: One of `train`, `dev`, and `test`. (default: `train`)
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance. (default: `ranking`)
    :param target_label: If `ranking`, chose one of `entailment`,
        `contradiction`, `neutral`, and `-` as the positive label.
        (default: `entailment`)
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.

    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    data_pack = _read_data(stage)

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        binary = (data_pack.relation['label']).astype(float)
        data_pack.relation['label'] = binary
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        label = data_pack.relation['label']
        data_pack.relation['label'] = label
        data_pack.one_hot_encode_label(num_classes=2, inplace=True)
        return data_pack
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")


def _read_data(stage):
    table = pd.read_csv(get_file_path(stage), sep='\t')
    df = pd.DataFrame({
        'text_left': table['sentence1'],
        'text_right': table['sentence2'],
        'label': table['gold_label']
    })
    df = df.dropna(axis=0, how='any').reset_index(drop=True)
    return matchzoo.pack(df)
