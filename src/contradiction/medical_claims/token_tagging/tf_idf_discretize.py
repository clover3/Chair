def idf_to_category_val(idf_v):
    if idf_v < 2.8:
        return 2
        # return "Stopword"
    elif idf_v < 6:
        return 6
        # return "Daily"
    elif idf_v < 9:
        return 9
        # return "Topical"
    elif idf_v < 17:
        return 17
        # return "Rare"
    else:
        return 18
        # return "OOV"


def idf_to_category_val_norm1(idf_v):
    return idf_to_category_val(idf_v) / 18


def daily_topic_rare(idf_v):
    if idf_v < 6:
        return 6
        # return "Daily"
    elif idf_v < 9:
        return 9
        # return "Topical"
    elif idf_v < 17:
        return 17
        # return "Rare"
    else:
        return 18
        # return "OOV"


def rare_vs_remain(idf_v):
    if idf_v < 9:
        return 9
        # return "Topical"
    elif idf_v < 17:
        return 17
        # return "Rare"
    else:
        return 9
        # return "OOV"


def category_stopword_oov(idf_v):
    if idf_v < 2.8:
        return 2
        # return "Stopword"
    else:
        return 18
        # return "OOV"


def category_stopword_daily_remain(idf_v):
    if idf_v < 2.8:
        return 2
        # return "Stopword"
    elif idf_v < 6:
        return 6
        # return "Daily"
    else:
        return 18
        # return "OOV"


def daily_remain(idf_v):
    if idf_v < 6:
        return 6
        # return "Daily"
    elif idf_v < 17:
        return 17
        # return "Rare"
    else:
        return 10
        # return "OOV"