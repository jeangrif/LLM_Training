def exact_match(pred, gold):
    pred = pred.lower().strip()
    gold = gold.lower().strip()
    return float(gold in pred)
