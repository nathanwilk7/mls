def print_eval_get_pct(correct, mistakes, do_print=True):
    correct_pct = (float(correct) / (correct + mistakes)) * 100.0
    if do_print:
        print('correct:', correct)
        print('mistakes:', mistakes)
        print('percent:', correct_pct)
    return correct_pct

def evaluate(eval_x, eval_y, eval_preds, label, do_print=True):
    if do_print:
        print(label)
    correct, mistakes = 0, 0
    assert len(eval_x) == len(eval_y) and len(eval_y) == len(eval_preds)
    for i in range(len(eval_preds)):
        if eval_preds[i] * eval_y[i] > 0:
            correct += 1
        else:
            mistakes += 1
    correct_pct = print_eval_get_pct(correct, mistakes, do_print=do_print)
    return correct, mistakes, correct_pct
