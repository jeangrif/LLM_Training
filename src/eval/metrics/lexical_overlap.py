def lexical_overlap(pred, gold):
    set_pred = set(pred.lower().split())
    set_gold = set(gold.lower().split())
    return len(set_pred & set_gold) / len(set_gold) if set_gold else 0
