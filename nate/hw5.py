import pdb

import numpy as np

from cross_validation import cross_val, data_from_file, get_xy, append_bias, add_any_missing_features_cols
from evaluation import evaluate, print_eval_get_pct
from svm_alg import svm
from log_reg_alg import log_reg
from bayes_alg import bayes

epochs = 1
folds = 5
cv_filepath = 'data/past_matches_train_cv_{cv}.csv' #'Dataset/CVSplits/training0{cv}.data'
initial_weight_range = 0.02
train_filepath = 'data/past_matches_train.csv'
test_filepath = 'data/past_matches_test.csv'
np.random.seed(7)

svm_learning_rates = [0.001]#[10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]
svm_regularizations = [0.001]#[10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]
log_reg_learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
log_reg_tradeoffs = [1.0, 10.0, 100.0, 1000.0, 10000.0]
bayes_smoothing = [2.0, 1.5, 1.0, 0.5]

features_to_drop = ['away_result', 'score_home', 'score_away', 'outcome', 'home_team_name', 'away_team_name', 'competition_id', 'season_id']
y_col = 'home_result'
num_features = 389

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

def hw5_wrapper(alg_label):
    # SVM
    svm_info = {}
    svm_info['learning_rate'] = None
    svm_info['regularization'] = None
    svm_info['w'] = None
    svm_info['cv_accuracy'] = 0
    svm_info['training_accuracy'] = 0
    svm_info['test_accuracy'] = 0

    for learning_rate in svm_learning_rates:
        for svm_regularization in svm_regularizations:
            w = np.random.rand(num_features + 1, 1) * initial_weight_range - (initial_weight_range / 2.0)
            for epoch in range(epochs):
                correct, mistakes = 0, 0
                for x, y, eval_x, eval_y in cross_val(folds, cv_filepath, num_features=num_features, is_svmlight=False, y_col=y_col, features_to_drop=features_to_drop):
                    #x, y = permute_examples(x, y)
                    w = svm(x, y, w, rate=learning_rate, regularization=svm_regularization)
                    temp_correct, temp_mistakes, _ = evaluate(eval_x, eval_y, eval_x.dot(w), 'EPOCH: '+str(epoch + 1), do_print=False)
                    correct += temp_correct
                    mistakes += temp_mistakes
            pct = print_eval_get_pct(correct, mistakes, do_print=False)
            if pct > svm_info['cv_accuracy']:
                svm_info['cv_accuracy'] = pct
                svm_info['learning_rate'] = learning_rate
                svm_info['regularization'] = svm_regularization
                svm_info['w'] = w
    
    # train set testing
    is_svmlight = False
    train_x, train_y = get_xy(data_from_file(train_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight, features_to_drop=features_to_drop, y_col=y_col)
    train_x = add_any_missing_features_cols(train_x, num_features)
    train_x = append_bias(train_x)
    pdb.set_trace()
    temp_correct, temp_mistakes, train_pct = evaluate(train_x, train_y, train_x.dot(svm_info['w']), 'train: ', do_print=False)
    svm_info['training_accuracy'] = train_pct

    is_svmlight = False
    test_x, test_y = get_xy(data_from_file(test_filepath, is_svmlight=is_svmlight), features_to_drop=features_to_drop, is_svmlight=is_svmlight, y_col=y_col)
    test_x = add_any_missing_features_cols(test_x, num_features)
    test_x = append_bias(test_x)
    temp_correct, temp_mistakes, test_pct = evaluate(test_x, test_y, test_x.dot(svm_info['w']), 'test: ', do_print=False)
    svm_info['test_accuracy'] = test_pct

    # # Logistic Regression
    # log_reg_info = {}
    # log_reg_info['learning_rate'] = None
    # log_reg_info['tradeoff'] = None
    # log_reg_info['w'] = None
    # log_reg_info['cv_accuracy'] = 0
    # log_reg_info['training_accuracy'] = 0
    # log_reg_info['test_accuracy'] = 0

    # for learning_rate in log_reg_learning_rates:
    #     for tradeoff in log_reg_tradeoffs:
    #         w = np.random.rand(num_features + 1, 1) * initial_weight_range - (initial_weight_range / 2.0)
    #         for epoch in range(epochs):
    #             correct, mistakes = 0, 0
    #             for x, y, eval_x, eval_y in cross_val(folds, cv_filepath, num_features=num_features):
    #                 x, y = permute_examples(x, y)
    #                 w = log_reg(x, y, w, rate=learning_rate, tradeoff=tradeoff)
    #                 temp_correct, temp_mistakes, _ = evaluate(eval_x, eval_y, eval_x * w, 'EPOCH: '+str(epoch + 1), do_print=False)
    #                 correct += temp_correct
    #                 mistakes += temp_mistakes
    #         pct = print_eval_get_pct(correct, mistakes, do_print=False)
    #         if pct > log_reg_info['cv_accuracy']:
    #             log_reg_info['cv_accuracy'] = pct
    #             log_reg_info['learning_rate'] = learning_rate
    #             log_reg_info['tradeoff'] = tradeoff
    #             log_reg_info['w'] = w
    
    # # train set testing
    # is_svmlight = True
    # train_x, train_y = get_xy(data_from_file(train_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight)
    # train_x = add_any_missing_features_cols(train_x, num_features)
    # train_x = append_bias(train_x)
    # temp_correct, temp_mistakes, train_pct = evaluate(train_x, train_y, train_x * log_reg_info['w'], 'train: ', do_print=False)
    # log_reg_info['training_accuracy'] = train_pct

    # is_svmlight = True
    # test_x, test_y = get_xy(data_from_file(test_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight)
    # test_x = add_any_missing_features_cols(test_x, num_features)
    # test_x = append_bias(test_x)
    # temp_correct, temp_mistakes, test_pct = evaluate(test_x, test_y, test_x * log_reg_info['w'], 'test: ', do_print=False)
    # log_reg_info['test_accuracy'] = test_pct

    # Naive Bayes
    # bayes_info = {}
    # bayes_info['smoothing'] = None
    # bayes_info['b'] = None
    # bayes_info['cv_accuracy'] = 0
    # bayes_info['training_accuracy'] = 0
    # bayes_info['test_accuracy'] = 0

    # for smoothing in bayes_smoothing:
    #     correct, mistakes = 0, 0
    #     for x, y, eval_x, eval_y in cross_val(folds, cv_filepath, num_features=num_features):
    #         b = bayes(x, y, smoothing=smoothing)
    #         preds = []
    #         for i in range(len(eval_x)):
    #             preds.append(b.predict(eval_x, i))
    #         temp_correct, temp_mistakes, _ = evaluate(eval_x, eval_y, preds, 'EPOCH: ', do_print=False)
    #         correct += temp_correct
    #         mistakes += temp_mistakes
    #     pct = print_eval_get_pct(correct, mistakes, do_print=False)
    #     if pct > bayes_info['cv_accuracy']:
    #         bayes_info['cv_accuracy'] = pct
    #         bayes_info['smoothing'] = smoothing
    #         bayes_info['b'] = b
    
    # # train set testing
    # is_svmlight = True
    # train_x, train_y = get_xy(data_from_file(train_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight)
    # train_x = add_any_missing_features_cols(train_x, num_features)
    # train_x = append_bias(train_x)
    # train_preds = []
    # for i in range(len(train_x)):
    #     train_preds.append(bayes_info['b'].predict(train_x, i))
    # temp_correct, temp_mistakes, train_pct = evaluate(train_x, train_y, train_preds, 'EPOCH: ', do_print=False)
    # bayes_info['training_accuracy'] = train_pct

    # is_svmlight = True
    # test_x, test_y = get_xy(data_from_file(test_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight)
    # test_x = add_any_missing_features_cols(test_x, num_features)
    # test_x = append_bias(test_x)
    # test_preds = []
    # for i in range(len(test_x)):
    #     test_preds.append(bayes_info['b'].predict(test_x, i))
    # temp_correct, temp_mistakes, test_pct = evaluate(test_x, test_y, test_preds, 'test: ', do_print=False)
    # bayes_info['test_accuracy'] = test_pct

    print('SVM')
    print('best hyperparameters:')
    print('- learning rate', svm_info['learning_rate'])
    print('- regularization', svm_info['regularization'])
    print('cv accuracy', svm_info['cv_accuracy'])
    print('training accuracy', svm_info['training_accuracy'])
    print('test accuracy', svm_info['test_accuracy'])
    print()

    # print('Logistic Regression')
    # print('best hyperparameters:')
    # print('- learning rate', log_reg_info['learning_rate'])
    # print('- tradeoff', log_reg_info['tradeoff'])
    # print('cv accuracy', log_reg_info['cv_accuracy'])
    # print('training accuracy', log_reg_info['training_accuracy'])
    # print('test accuracy', log_reg_info['test_accuracy'])
    # print()

    # print('Naive Bayes')
    # print('best hyperparameter:')
    # print('- smoothing', bayes_info['smoothing'])
    # print('cv accuracy', bayes_info['cv_accuracy'])
    # print('training accuracy', bayes_info['training_accuracy'])
    # print('test accuracy', bayes_info['test_accuracy'])
    # print()

hw5_wrapper('')
