import pdb

import numpy as np

from cross_validation import cross_val, data_from_file, get_xy, append_bias, add_any_missing_features_cols
from evaluation import evaluate, print_eval_get_pct
from perceptron_alg import perceptron

epochs = 10
folds = 5
cv_filepath = 'Dataset/CVSplits/training0{cv}.data'
initial_weight_range=0.02
num_features = 68
learning_rates = [1, 0.1, 0.01]
margins = [1, 0.1, 0.01]
dev_filepath = 'Dataset/phishing.dev'
test_filepath = 'Dataset/phishing.test'
dev_epochs = 20
np.random.seed(7)

def permute_examples(x, y):
    shuffle_is = np.arange(x.shape[0])
    np.random.shuffle(shuffle_is)
    temp_x = np.empty(x.shape)
    temp_y = np.empty(y.shape)
    counter = 0
    for shuffle_i in shuffle_is:
        temp_x[counter] = x[shuffle_i]
        temp_y[counter] = y[shuffle_i]
        counter += 1
    return temp_x, temp_y

# margin perceptron
def perceptron_wrapper(alg_label, learning_rates=[1], dynamic_learning_rate_epochs=False, dynamic_learning_rate_examples=False,
                       margins=[0], averaged=False):
    training_updates = 0
    best_alg = {}
    best_alg['label'] = None
    best_alg['cv_pct'] = 0
    best_alg['dev_pct'] = 0
    best_alg['test_pct'] = 0
    best_alg['dyn_epochs'] = None
    best_alg['dyn_examples'] = None
    best_alg['avg'] = None
    best_alg['margin'] = None
    best_alg['rate'] = None
    best_alg['w'] = None
    
    for learning_rate in learning_rates:
        initial_learning_rate = learning_rate
        if dynamic_learning_rate_epochs or dynamic_learning_rate_examples:
            dynamic_learning_rate_num = 0
        for margin in margins:
            print('{} learning rate:'.format(alg_label), learning_rate)
            if margins != [0]:
                print('{} margin:'.format(alg_label), margin)
                pass
            w = np.random.rand(num_features + 1, 1) * initial_weight_range - (initial_weight_range / 2.0)
            if averaged:
                running_avg = np.zeros((num_features + 1, 1))
            else:
                running_avg = None
            for epoch in range(epochs):
                correct, mistakes = 0, 0
                if dynamic_learning_rate_epochs:
                    learning_rate = initial_learning_rate / (1 + epoch)
                for x, y, eval_x, eval_y in cross_val(folds, cv_filepath, num_features=num_features):
                    # permute instances
                    x, y = permute_examples(x, y)
                    # initialize random initial weights and bias
                    if dynamic_learning_rate_examples:
                        learning_rate = initial_learning_rate / (1 + dynamic_learning_rate_num)
                        w, temp_training_updates = perceptron(x, y, w, rate=learning_rate, dynamic_learning_num=dynamic_learning_rate_num, initial_learning_rate=initial_learning_rate)
                        dynamic_learning_rate_num += x.shape[0]
                    else:
                        w, temp_training_updates = perceptron(x, y, w, rate=learning_rate, margin=margin, average=running_avg)
                    training_updates += temp_training_updates
                    temp_correct, temp_mistakes, _ = evaluate(eval_x, eval_y, eval_x * w, 'EPOCH: '+str(epoch + 1), do_print=False)
                    correct += temp_correct
                    mistakes += temp_mistakes
                print('epoch:', epoch + 1)
                print_eval_get_pct(correct, mistakes)
            print()
            print('final {} learning rate:'.format(alg_label), learning_rate)
            if margins != [0]:
                print('final {} margin:'.format(alg_label), margin)
                pass
            pct = print_eval_get_pct(correct, mistakes, do_print=False)
            if pct > best_alg['cv_pct']:
                best_alg['label'] = alg_label
                best_alg['cv_pct'] = pct
                best_alg['dyn_epochs'] = dynamic_learning_rate_epochs
                best_alg['dyn_examples'] = dynamic_learning_rate_examples
                best_alg['avg'] = averaged
                best_alg['margin'] = margin
                best_alg['rate'] = initial_learning_rate
                best_alg['w'] = w
            print()
    
    # dev set testing
    is_svmlight = True
    dev_x, dev_y = get_xy(data_from_file(dev_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight)
    dev_x = add_any_missing_features_cols(dev_x, num_features)
    dev_x = append_bias(dev_x)
    best_alg['dev_plot_y'] = []
    for epoch in range(dev_epochs):
        correct, mistakes = 0, 0
        if dynamic_learning_rate_epochs:
            learning_rate = initial_learning_rate / (1 + epoch)
        for x, y, _, _ in cross_val(folds, cv_filepath, num_features=num_features, leave_out_none=True):
            # permute instances
            x, y = permute_examples(x, y)
            # initialize random initial weights and bias
            if dynamic_learning_rate_examples:
                learning_rate = initial_learning_rate / (1 + dynamic_learning_rate_num)
                w, temp_training_updates = perceptron(x, y, w, rate=learning_rate, dynamic_learning_num=dynamic_learning_rate_num, initial_learning_rate=initial_learning_rate)
                dynamic_learning_rate_num += x.shape[0]
            else:
                w, temp_training_updates = perceptron(x, y, w, rate=learning_rate, margin=margin, average=running_avg)
            training_updates += temp_training_updates
            temp_correct, temp_mistakes, _ = evaluate(dev_x, dev_y, dev_x * w, 'dev epoch: '+str(epoch + 1), do_print=False)
            correct += temp_correct
            mistakes += temp_mistakes
        print('dev epoch:', epoch + 1)
        pct = print_eval_get_pct(correct, mistakes)
        best_alg['dev_plot_y'].append(pct)
        if pct > best_alg['dev_pct']:
            best_alg['label'] = alg_label
            best_alg['dev_pct'] = pct
            best_alg['dyn_epochs'] = dynamic_learning_rate_epochs
            best_alg['dyn_examples'] = dynamic_learning_rate_examples
            best_alg['avg'] = averaged
            best_alg['margin'] = margin
            best_alg['rate'] = initial_learning_rate
            best_alg['w'] = w
            best_alg['epoch'] = epoch + 1

    is_svmlight = True
    test_x, test_y = get_xy(data_from_file(dev_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight)
    test_x = add_any_missing_features_cols(test_x, num_features)
    test_x = append_bias(test_x)
    temp_correct, temp_mistakes, test_pct = evaluate(test_x, test_y, test_x * best_alg['w'], 'test: ')
    best_alg['test_pct'] = test_pct
    # to make printing cleaner
    best_alg['w'] = None
    print(best_alg)
    print('training updates:', training_updates)

perceptron_wrapper('simple perceptron', learning_rates=learning_rates)
perceptron_wrapper('dynamic learning rate perceptron (on epochs)', learning_rates=learning_rates, dynamic_learning_rate_epochs=True)
#perceptron_wrapper('dynamic learning rate perceptron (on examples)', learning_rates=learning_rates, dynamic_learning_rate_examples=True)
perceptron_wrapper('margin perceptron', learning_rates=learning_rates, margins=margins, dynamic_learning_rate_epochs=True)
perceptron_wrapper('averaged perceptron', learning_rates=learning_rates, averaged=True)
