def to_json(problem_id, seg_instance, score):
    return {
        'problem_id': problem_id,
        'seg_instance': seg_instance.to_json(),
        'score': score
    }