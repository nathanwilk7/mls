import pdb

import numpy as np

from cross_validation import cross_val, data_from_file, get_xy, append_bias, add_any_missing_features_cols
from evaluation import evaluate, print_eval_get_pct
from svm_alg import svm
from log_reg_alg import log_reg
from bayes_alg import bayes

epochs = 1
folds = 5
#cv_filepath = 'data/past_matches_results_train_cv_{cv}.csv' #'Dataset/CVSplits/training0{cv}.data'
cv_filepath = 'data/mls_16_train_cv_{cv}.csv' #'Dataset/CVSplits/training0{cv}.data'
initial_weight_range = 0.02
train_filepath = 'data/mls_16_train.csv'
test_filepath = 'data/mls_16_test.csv'
np.random.seed(7)

svm_learning_rates = [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]
svm_regularizations = [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]
log_reg_learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
log_reg_tradeoffs = [1.0, 10.0, 100.0, 1000.0, 10000.0]
#bayes_smoothing = [2.0]#, 1.5, 1.0, 0.5]

home_features_to_drop = ['score_home', 'score_away', 'outcome', 'home_team_name', 'away_team_name', 'competition_id', 'season_id', 'home_half_score_home', 'home_half_score_away', 'away_half_score_home', 'away_half_score_away', 'away_result']
away_features_to_drop = ['score_home', 'score_away', 'outcome', 'home_team_name', 'away_team_name', 'competition_id', 'season_id', 'home_half_score_home', 'home_half_score_away', 'away_half_score_home', 'away_half_score_away', 'home_result']
# features_to_drop = [#'away_result',
#                     'score_home',
#                     'score_away',
#                     #'outcome',
#                     #'home_team_name',
#                     #'away_team_name',
#                     #'competition_id',
#                     #'season_id',
#                     'home_score_home',
#                     'home_score_away',
#                     'home_half_score_home',
#                     'home_half_score_away',
#                     'home_away_equalizer_goal',
#                     'home_away_matchwinner_goal',
#                     'home_away_outofplayconceded_f24',
#                     'home_away_savesbody_f24',
#                     'home_away_savescaught_f24',
#                     'home_away_savescollected_f24',
#                     'home_away_savesdiving_f24',
#                     'home_away_savesfeet_f24',
#                     'home_away_savesfingertip_f24',
#                     'home_away_saveshands_f24',
#                     'home_away_savesparrieddanger_f24',
#                     'home_away_savesparriedsafe_f24',
#                     'home_away_savesreaching_f24',
#                     'home_away_savesstanding_f24',
#                     'home_away_savesstooping_f24',
#                     'home_away_shotsclearedofflineinsidebox_f24',
#                     'home_away_shotsclearedofflineoutsidebox_f24',
#                     'home_away_accurate_back_zone_pass',
#                     'home_away_accurate_corners_intobox',
#                     'home_away_accurate_cross',
#                     'home_away_accurate_keeper_throws',
#                     'home_away_accurate_through_ball',
#                     'home_away_aerial_lost',
#                     'home_away_aerial_won',
#                     'home_away_att_hd_goal',
#                     'home_away_att_hd_off_target',
#                     'home_away_att_hd_target',
#                     'home_away_att_ibox_blocked',
#                     'home_away_att_ibox_goal',
#                     'home_away_att_ibox_off_target',
#                     'home_away_att_ibox_target',
#                     'home_away_att_obox_blocked',
#                     'home_away_att_obox_goal',
#                     'home_away_att_obox_off_target',
#                     'home_away_att_obox_target',
#                     'home_away_att_pen_goal',
#                     'home_away_att_pen_off_target',
#                     'home_away_att_pen_target',
#                     'home_away_ball_recovery',
#                     'home_away_challenge_lost',
#                     'home_away_clean_sheet',
#                     'home_away_clean_sheet_away',
#                     'home_away_clean_sheet_part',
#                     'home_away_clearance_off_line',
#                     'home_away_direct_goals',
#                     'home_away_drops',
#                     'home_away_error_lead_to_goal',
#                     'home_away_error_lead_to_shot',
#                     'home_away_failed_back_zone_pass',
#                     'home_away_failed_corners_intobox',
#                     'home_away_failed_cross',
#                     'home_away_failed_fwd_zone_pass',
#                     'home_away_failed_keeper_throws',
#                     'home_away_fouls',
#                     'home_away_gk_smother',
#                     'home_away_goal_assist',
#                     'home_away_goal_assist_openplay',
#                     'home_away_goals_conceded_ibox',
#                     'home_away_goals_conceded_obox',
#                     'home_away_good_high_claim',
#                     'home_away_hand_ball',
#                     'home_away_head_clearance',
#                     'home_away_hit_woodwork',
#                     'home_away_interception',
#                     'home_away_key_passes',
#                     'home_away_key_passes_open_play',
#                     'home_away_last_man_tackle',
#                     'home_away_long_pass_own_to_opp',
#                     'home_away_long_pass_own_to_opp_success',
#                     'home_away_lost_contest',
#                     'home_away_lost_corners',
#                     'home_away_lost_tackle',
#                     'home_away_nohead_clearance',
#                     'home_away_own_goals',
#                     'home_away_penalty_conceded',
#                     'home_away_penalty_save',
#                     'home_away_penalty_won',
#                     'home_away_punches',
#                     'home_away_red_card',
#                     'home_away_saved_ibox',
#                     'home_away_saved_obox',
#                     'home_away_second_yellow',
#                     'home_away_shield_ball_oop',
#                     'home_away_six_yard_block',
#                     'home_away_successful_final_third_passes',
#                     'home_away_total_attacking_pass',
#                     'home_away_total_fwd_zone_pass',
#                     'home_away_total_offside',
#                     'home_away_was_fouled',
#                     'home_away_won_contest',
#                     'home_away_won_corners',
#                     'home_away_won_tackle',
#                     'home_away_yellow_card',
#                     'home_home_equalizer_goal',
#                     'home_home_matchwinner_goal',
#                     'home_home_outofplayconceded_f24',
#                     'home_home_savesbody_f24',
#                     'home_home_savescaught_f24',
#                     'home_home_savescollected_f24',
#                     'home_home_savesdiving_f24',
#                     'home_home_savesfeet_f24',
#                     'home_home_savesfingertip_f24',
#                     'home_home_saveshands_f24',
#                     'home_home_savesparrieddanger_f24',
#                     'home_home_savesparriedsafe_f24',
#                     'home_home_savesreaching_f24',
#                     'home_home_savesstanding_f24',
#                     'home_home_savesstooping_f24',
#                     'home_home_shotsclearedofflineinsidebox_f24',
#                     'home_home_shotsclearedofflineoutsidebox_f24',
#                     'home_home_accurate_back_zone_pass',
#                     'home_home_accurate_corners_intobox',
#                     'home_home_accurate_cross',
#                     'home_home_accurate_keeper_throws',
#                     'home_home_accurate_through_ball',
#                     'home_home_aerial_lost',
#                     'home_home_aerial_won',
#                     'home_home_att_hd_goal',
#                     'home_home_att_hd_off_target',
#                     'home_home_att_hd_target',
#                     'home_home_att_ibox_blocked',
#                     'home_home_att_ibox_goal',
#                     'home_home_att_ibox_off_target',
#                     'home_home_att_ibox_target',
#                     'home_home_att_obox_blocked',
#                     'home_home_att_obox_goal',
#                     'home_home_att_obox_off_target',
#                     'home_home_att_obox_target',
#                     'home_home_att_pen_goal',
#                     'home_home_att_pen_off_target',
#                     'home_home_att_pen_target',
#                     'home_home_ball_recovery',
#                     'home_home_challenge_lost',
#                     'home_home_clean_sheet',
#                     'home_home_clean_sheet_away',
#                     'home_home_clean_sheet_part',
#                     'home_home_clearance_off_line',
#                     'home_home_direct_goals',
#                     'home_home_drops',
#                     'home_home_error_lead_to_goal',
#                     'home_home_error_lead_to_shot',
#                     'home_home_failed_back_zone_pass',
#                     'home_home_failed_corners_intobox',
#                     'home_home_failed_cross',
#                     'home_home_failed_fwd_zone_pass',
#                     'home_home_failed_keeper_throws',
#                     'home_home_fouls',
#                     'home_home_gk_smother',
#                     'home_home_goal_assist',
#                     'home_home_goal_assist_openplay',
#                     'home_home_goals_conceded_ibox',
#                     'home_home_goals_conceded_obox',
#                     'home_home_good_high_claim',
#                     'home_home_hand_ball',
#                     'home_home_head_clearance',
#                     'home_home_hit_woodwork',
#                     'home_home_interception',
#                     'home_home_key_passes',
#                     'home_home_key_passes_open_play',
#                     'home_home_last_man_tackle',
#                     'home_home_long_pass_own_to_opp',
#                     'home_home_long_pass_own_to_opp_success',
#                     'home_home_lost_contest',
#                     'home_home_lost_corners',
#                     'home_home_lost_tackle',
#                     'home_home_nohead_clearance',
#                     'home_home_own_goals',
#                     'home_home_penalty_conceded',
#                     'home_home_penalty_save',
#                     'home_home_penalty_won',
#                     'home_home_punches',
#                     'home_home_red_card',
#                     'home_home_saved_ibox',
#                     'home_home_saved_obox',
#                     'home_home_second_yellow',
#                     'home_home_shield_ball_oop',
#                     'home_home_six_yard_block',
#                     'home_home_successful_final_third_passes',
#                     'home_home_total_attacking_pass',
#                     'home_home_total_fwd_zone_pass',
#                     'home_home_total_offside',
#                     'home_home_was_fouled',
#                     'home_home_won_contest',
#                     'home_home_won_corners',
#                     'home_home_won_tackle',
#                     'home_home_yellow_card',
#                     #'home_team_score',
#                     #'home_opponent_score',
#                     #'home_past_wins',
#                     #'home_past_ties',
#                     #'home_past_losses',
#                     #'home_past_ppg',
#                     'away_score_home',
#                     'away_score_away',
#                     'away_half_score_home',
#                     'away_half_score_away',
#                     'away_away_equalizer_goal',
#                     'away_away_matchwinner_goal',
#                     'away_away_outofplayconceded_f24',
#                     'away_away_savesbody_f24',
#                     'away_away_savescaught_f24',
#                     'away_away_savescollected_f24',
#                     'away_away_savesdiving_f24',
#                     'away_away_savesfeet_f24',
#                     'away_away_savesfingertip_f24',
#                     'away_away_saveshands_f24',
#                     'away_away_savesparrieddanger_f24',
#                     'away_away_savesparriedsafe_f24',
#                     'away_away_savesreaching_f24',
#                     'away_away_savesstanding_f24',
#                     'away_away_savesstooping_f24',
#                     'away_away_shotsclearedofflineinsidebox_f24',
#                     'away_away_shotsclearedofflineoutsidebox_f24',
#                     'away_away_accurate_back_zone_pass',
#                     'away_away_accurate_corners_intobox',
#                     'away_away_accurate_cross',
#                     'away_away_accurate_keeper_throws',
#                     'away_away_accurate_through_ball',
#                     'away_away_aerial_lost',
#                     'away_away_aerial_won',
#                     'away_away_att_hd_goal',
#                     'away_away_att_hd_off_target',
#                     'away_away_att_hd_target',
#                     'away_away_att_ibox_blocked',
#                     'away_away_att_ibox_goal',
#                     'away_away_att_ibox_off_target',
#                     'away_away_att_ibox_target',
#                     'away_away_att_obox_blocked',
#                     'away_away_att_obox_goal',
#                     'away_away_att_obox_off_target',
#                     'away_away_att_obox_target',
#                     'away_away_att_pen_goal',
#                     'away_away_att_pen_off_target',
#                     'away_away_att_pen_target',
#                     'away_away_ball_recovery',
#                     'away_away_challenge_lost',
#                     'away_away_clean_sheet',
#                     'away_away_clean_sheet_away',
#                     'away_away_clean_sheet_part',
#                     'away_away_clearance_off_line',
#                     'away_away_direct_goals',
#                     'away_away_drops',
#                     'away_away_error_lead_to_goal',
#                     'away_away_error_lead_to_shot',
#                     'away_away_failed_back_zone_pass',
#                     'away_away_failed_corners_intobox',
#                     'away_away_failed_cross',
#                     'away_away_failed_fwd_zone_pass',
#                     'away_away_failed_keeper_throws',
#                     'away_away_fouls',
#                     'away_away_gk_smother',
#                     'away_away_goal_assist',
#                     'away_away_goal_assist_openplay',
#                     'away_away_goals_conceded_ibox',
#                     'away_away_goals_conceded_obox',
#                     'away_away_good_high_claim',
#                     'away_away_hand_ball',
#                     'away_away_head_clearance',
#                     'away_away_hit_woodwork',
#                     'away_away_interception',
#                     'away_away_key_passes',
#                     'away_away_key_passes_open_play',
#                     'away_away_last_man_tackle',
#                     'away_away_long_pass_own_to_opp',
#                     'away_away_long_pass_own_to_opp_success',
#                     'away_away_lost_contest',
#                     'away_away_lost_corners',
#                     'away_away_lost_tackle',
#                     'away_away_nohead_clearance',
#                     'away_away_own_goals',
#                     'away_away_penalty_conceded',
#                     'away_away_penalty_save',
#                     'away_away_penalty_won',
#                     'away_away_punches',
#                     'away_away_red_card',
#                     'away_away_saved_ibox',
#                     'away_away_saved_obox',
#                     'away_away_second_yellow',
#                     'away_away_shield_ball_oop',
#                     'away_away_six_yard_block',
#                     'away_away_successful_final_third_passes',
#                     'away_away_total_attacking_pass',
#                     'away_away_total_fwd_zone_pass',
#                     'away_away_total_offside',
#                     'away_away_was_fouled',
#                     'away_away_won_contest',
#                     'away_away_won_corners',
#                     'away_away_won_tackle',
#                     'away_away_yellow_card',
#                     'away_home_equalizer_goal',
#                     'away_home_matchwinner_goal',
#                     'away_home_outofplayconceded_f24',
#                     'away_home_savesbody_f24',
#                     'away_home_savescaught_f24',
#                     'away_home_savescollected_f24',
#                     'away_home_savesdiving_f24',
#                     'away_home_savesfeet_f24',
#                     'away_home_savesfingertip_f24',
#                     'away_home_saveshands_f24',
#                     'away_home_savesparrieddanger_f24',
#                     'away_home_savesparriedsafe_f24',
#                     'away_home_savesreaching_f24',
#                     'away_home_savesstanding_f24',
#                     'away_home_savesstooping_f24',
#                     'away_home_shotsclearedofflineinsidebox_f24',
#                     'away_home_shotsclearedofflineoutsidebox_f24',
#                     'away_home_accurate_back_zone_pass',
#                     'away_home_accurate_corners_intobox',
#                     'away_home_accurate_cross',
#                     'away_home_accurate_keeper_throws',
#                     'away_home_accurate_through_ball',
#                     'away_home_aerial_lost',
#                     'away_home_aerial_won',
#                     'away_home_att_hd_goal',
#                     'away_home_att_hd_off_target',
#                     'away_home_att_hd_target',
#                     'away_home_att_ibox_blocked',
#                     'away_home_att_ibox_goal',
#                     'away_home_att_ibox_off_target',
#                     'away_home_att_ibox_target',
#                     'away_home_att_obox_blocked',
#                     'away_home_att_obox_goal',
#                     'away_home_att_obox_off_target',
#                     'away_home_att_obox_target',
#                     'away_home_att_pen_goal',
#                     'away_home_att_pen_off_target',
#                     'away_home_att_pen_target',
#                     'away_home_ball_recovery',
#                     'away_home_challenge_lost',
#                     'away_home_clean_sheet',
#                     'away_home_clean_sheet_away',
#                     'away_home_clean_sheet_part',
#                     'away_home_clearance_off_line',
#                     'away_home_direct_goals',
#                     'away_home_drops',
#                     'away_home_error_lead_to_goal',
#                     'away_home_error_lead_to_shot',
#                     'away_home_failed_back_zone_pass',
#                     'away_home_failed_corners_intobox',
#                     'away_home_failed_cross',
#                     'away_home_failed_fwd_zone_pass',
#                     'away_home_failed_keeper_throws',
#                     'away_home_fouls',
#                     'away_home_gk_smother',
#                     'away_home_goal_assist',
#                     'away_home_goal_assist_openplay',
#                     'away_home_goals_conceded_ibox',
#                     'away_home_goals_conceded_obox',
#                     'away_home_good_high_claim',
#                     'away_home_hand_ball',
#                     'away_home_head_clearance',
#                     'away_home_hit_woodwork',
#                     'away_home_interception',
#                     'away_home_key_passes',
#                     'away_home_key_passes_open_play',
#                     'away_home_last_man_tackle',
#                     'away_home_long_pass_own_to_opp',
#                     'away_home_long_pass_own_to_opp_success',
#                     'away_home_lost_contest',
#                     'away_home_lost_corners',
#                     'away_home_lost_tackle',
#                     'away_home_nohead_clearance',
#                     'away_home_own_goals',
#                     'away_home_penalty_conceded',
#                     'away_home_penalty_save',
#                     'away_home_penalty_won',
#                     'away_home_punches',
#                     'away_home_red_card',
#                     'away_home_saved_ibox',
#                     'away_home_saved_obox',
#                     'away_home_second_yellow',
#                     'away_home_shield_ball_oop',
#                     'away_home_six_yard_block',
#                     'away_home_successful_final_third_passes',
#                     'away_home_total_attacking_pass',
#                     'away_home_total_fwd_zone_pass',
#                     'away_home_total_offside',
#                     'away_home_was_fouled',
#                     'away_home_won_contest',
#                     'away_home_won_corners',
#                     'away_home_won_tackle',
#                     'away_home_yellow_card',
#                     #'away_team_score',
#                     #'away_opponent_score',
#                     #'away_past_wins',
#                     #'away_past_ties',
#                     #'away_past_losses',
#                     #'away_past_ppg',
#                     'outcome',
#                     #'home_result',
#                     'away_result',
#                     'date',
#                     'match_id',
#                     'home_team_name',
#                     'away_team_name',
#                     'competition_id',
#                     'season_id',
#                     #'score_home',
#                     #'score_away'
# ]
home_y_col = 'home_result'
away_y_col = 'away_result'
num_features = 393 # 15

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
    is_svmlight = False

    # SVM home
    svm_info_home = {}
    svm_info_home['learning_rate'] = None
    svm_info_home['regularization'] = None
    svm_info_home['w'] = None
    svm_info_home['cv_accuracy'] = 0
    svm_info_home['training_accuracy'] = 0
    svm_info_home['test_accuracy'] = 0
    svm_info_home['win_tie_loss_accuracy'] = 0.0

    for learning_rate in svm_learning_rates:
        for svm_regularization in svm_regularizations:
            print('svm home', learning_rate, 'of', svm_learning_rates, svm_regularization, 'of', svm_regularizations)
            w = np.random.rand(num_features + 1, 1) * initial_weight_range - (initial_weight_range / 2.0)
            for epoch in range(epochs):
                correct, mistakes = 0, 0
                for x, y, eval_x, eval_y in cross_val(folds, cv_filepath, num_features=num_features, is_svmlight=False, y_col=home_y_col, features_to_drop=home_features_to_drop):
                    #x, y = permute_examples(x, y)
                    w = svm(x, y, w, rate=learning_rate, regularization=svm_regularization)
                    temp_correct, temp_mistakes, _ = evaluate(eval_x, eval_y, eval_x.dot(w), 'EPOCH: '+str(epoch + 1), do_print=False)
                    correct += temp_correct
                    mistakes += temp_mistakes
            pct = print_eval_get_pct(correct, mistakes, do_print=False)
            if pct > svm_info_home['cv_accuracy']:
                svm_info_home['cv_accuracy'] = pct
                svm_info_home['learning_rate'] = learning_rate
                svm_info_home['regularization'] = svm_regularization
                svm_info_home['w'] = w
    
    # train set testing
    is_svmlight = False
    train_x, train_y = get_xy(data_from_file(train_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight, features_to_drop=home_features_to_drop, y_col=home_y_col)
    train_x = add_any_missing_features_cols(train_x, num_features)
    train_x = append_bias(train_x)
    temp_correct, temp_mistakes, train_pct = evaluate(train_x, train_y, train_x.dot(svm_info_home['w']), 'train: ', do_print=False)
    svm_info_home['training_accuracy'] = train_pct

    is_svmlight = False
    test_x, test_y = get_xy(data_from_file(test_filepath, is_svmlight=is_svmlight), features_to_drop=home_features_to_drop, is_svmlight=is_svmlight, y_col=home_y_col)
    home_test_y = test_y
    test_x = add_any_missing_features_cols(test_x, num_features)
    test_x = append_bias(test_x)
    temp_correct, temp_mistakes, test_pct = evaluate(test_x, test_y, test_x.dot(svm_info_home['w']), 'test: ', do_print=False)
    svm_info_home['test_accuracy'] = test_pct

    # SVM away
    svm_info_away = {}
    svm_info_away['learning_rate'] = None
    svm_info_away['regularization'] = None
    svm_info_away['w'] = None
    svm_info_away['cv_accuracy'] = 0
    svm_info_away['training_accuracy'] = 0
    svm_info_away['test_accuracy'] = 0
    svm_info_away['win_tie_loss_accuracy'] = 0.0

    for learning_rate in svm_learning_rates:
        for svm_regularization in svm_regularizations:
            print('svm away', learning_rate, 'of', svm_learning_rates, svm_regularization, 'of', svm_regularizations)
            w = np.random.rand(num_features + 1, 1) * initial_weight_range - (initial_weight_range / 2.0)
            for epoch in range(epochs):
                correct, mistakes = 0, 0
                for x, y, eval_x, eval_y in cross_val(folds, cv_filepath, num_features=num_features, is_svmlight=False, y_col=away_y_col, features_to_drop=away_features_to_drop):
                    #x, y = permute_examples(x, y)
                    w = svm(x, y, w, rate=learning_rate, regularization=svm_regularization)
                    temp_correct, temp_mistakes, _ = evaluate(eval_x, eval_y, eval_x.dot(w), 'EPOCH: '+str(epoch + 1), do_print=False)
                    correct += temp_correct
                    mistakes += temp_mistakes
            pct = print_eval_get_pct(correct, mistakes, do_print=False)
            if pct > svm_info_away['cv_accuracy']:
                svm_info_away['cv_accuracy'] = pct
                svm_info_away['learning_rate'] = learning_rate
                svm_info_away['regularization'] = svm_regularization
                svm_info_away['w'] = w
    
    # train set testing
    is_svmlight = False
    train_x, train_y = get_xy(data_from_file(train_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight, features_to_drop=away_features_to_drop, y_col=away_y_col)
    train_x = add_any_missing_features_cols(train_x, num_features)
    train_x = append_bias(train_x)
    temp_correct, temp_mistakes, train_pct = evaluate(train_x, train_y, train_x.dot(svm_info_away['w']), 'train: ', do_print=False)
    svm_info_away['training_accuracy'] = train_pct

    is_svmlight = False
    test_x, test_y = get_xy(data_from_file(test_filepath, is_svmlight=is_svmlight), features_to_drop=away_features_to_drop, is_svmlight=is_svmlight, y_col=away_y_col)
    away_test_y = test_y
    test_x = add_any_missing_features_cols(test_x, num_features)
    test_x = append_bias(test_x)
    temp_correct, temp_mistakes, test_pct = evaluate(test_x, test_y, test_x.dot(svm_info_away['w']), 'test: ', do_print=False)
    svm_info_away['test_accuracy'] = test_pct

    def win_tie_loss_pct(home_w, away_w, x, home_y, away_y):
        home_preds = x.dot(home_w)
        away_preds = x.dot(away_w)
        correct, mistakes = 0.0, 0.0
        for i in range(len(home_preds)):
            if home_preds[i] * home_y[i] > 0 and away_preds[i] * away_y[i] > 0:
                correct += 1
            else:
                mistakes += 1
        return (correct / float(correct + mistakes)) * 100

    pdb.set_trace()
    svm_info_home['win_tie_loss_accuracy'] = win_tie_loss_pct(svm_info_home['w'], svm_info_away['w'], test_x, home_test_y, away_test_y)
    svm_info_away['win_tie_loss_accuracy'] = svm_info_home['win_tie_loss_accuracy']

    # log reg home
    log_reg_info_home = {}
    log_reg_info_home['learning_rate'] = None
    log_reg_info_home['tradeoff'] = None
    log_reg_info_home['w'] = None
    log_reg_info_home['cv_accuracy'] = 0
    log_reg_info_home['training_accuracy'] = 0
    log_reg_info_home['test_accuracy'] = 0
    log_reg_info_home['win_tie_loss_accuracy'] = 0.0

    for learning_rate in log_reg_learning_rates:
        for tradeoff in log_reg_tradeoffs:
            print('log reg home', learning_rate, 'of', log_reg_learning_rates, tradeoff, 'of', log_reg_tradeoffs)
            w = np.random.rand(num_features + 1, 1) * initial_weight_range - (initial_weight_range / 2.0)
            for epoch in range(epochs):
                correct, mistakes = 0, 0
                for x, y, eval_x, eval_y in cross_val(folds, cv_filepath, num_features=num_features, is_svmlight=False, y_col=home_y_col, features_to_drop=home_features_to_drop):
                    #x, y = permute_examples(x, y)
                    w = log_reg(x, y, w, rate=learning_rate, tradeoff=tradeoff)
                    temp_correct, temp_mistakes, _ = evaluate(eval_x, eval_y, eval_x.dot(w), 'EPOCH: '+str(epoch + 1), do_print=False)
                    correct += temp_correct
                    mistakes += temp_mistakes
            pct = print_eval_get_pct(correct, mistakes, do_print=False)
            if pct > log_reg_info_home['cv_accuracy']:
                log_reg_info_home['cv_accuracy'] = pct
                log_reg_info_home['learning_rate'] = learning_rate
                log_reg_info_home['tradeoff'] = tradeoff
                log_reg_info_home['w'] = w
    
    # train set testing
    is_svmlight = False
    train_x, train_y = get_xy(data_from_file(train_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight, features_to_drop=home_features_to_drop, y_col=home_y_col)
    train_x = add_any_missing_features_cols(train_x, num_features)
    train_x = append_bias(train_x)
    temp_correct, temp_mistakes, train_pct = evaluate(train_x, train_y, train_x.dot(log_reg_info_home['w']), 'train: ', do_print=False)
    log_reg_info_home['training_accuracy'] = train_pct

    is_svmlight = False
    test_x, test_y = get_xy(data_from_file(test_filepath, is_svmlight=is_svmlight), features_to_drop=home_features_to_drop, is_svmlight=is_svmlight, y_col=home_y_col)
    home_test_y = test_y
    test_x = add_any_missing_features_cols(test_x, num_features)
    test_x = append_bias(test_x)
    temp_correct, temp_mistakes, test_pct = evaluate(test_x, test_y, test_x.dot(log_reg_info_home['w']), 'test: ', do_print=False)
    log_reg_info_home['test_accuracy'] = test_pct

    # SVM away
    log_reg_info_away = {}
    log_reg_info_away['learning_rate'] = None
    log_reg_info_away['tradeoff'] = None
    log_reg_info_away['w'] = None
    log_reg_info_away['cv_accuracy'] = 0
    log_reg_info_away['training_accuracy'] = 0
    log_reg_info_away['test_accuracy'] = 0
    log_reg_info_away['win_tie_loss_accuracy'] = 0.0

    for learning_rate in log_reg_learning_rates:
        for tradeoff in log_reg_tradeoffs:
            print('log reg away', learning_rate, 'of', log_reg_learning_rates, tradeoff, 'of', log_reg_tradeoffs)
            w = np.random.rand(num_features + 1, 1) * initial_weight_range - (initial_weight_range / 2.0)
            for epoch in range(epochs):
                correct, mistakes = 0, 0
                for x, y, eval_x, eval_y in cross_val(folds, cv_filepath, num_features=num_features, is_svmlight=False, y_col=away_y_col, features_to_drop=away_features_to_drop):
                    #x, y = permute_examples(x, y)
                    w = log_reg(x, y, w, rate=learning_rate, tradeoff=tradeoff)
                    temp_correct, temp_mistakes, _ = evaluate(eval_x, eval_y, eval_x.dot(w), 'EPOCH: '+str(epoch + 1), do_print=False)
                    correct += temp_correct
                    mistakes += temp_mistakes
            pct = print_eval_get_pct(correct, mistakes, do_print=False)
            if pct > log_reg_info_away['cv_accuracy']:
                log_reg_info_away['cv_accuracy'] = pct
                log_reg_info_away['learning_rate'] = learning_rate
                log_reg_info_away['tradeoff'] = tradeoff
                log_reg_info_away['w'] = w
    
    # train set testing
    is_svmlight = False
    train_x, train_y = get_xy(data_from_file(train_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight, features_to_drop=away_features_to_drop, y_col=away_y_col)
    train_x = add_any_missing_features_cols(train_x, num_features)
    train_x = append_bias(train_x)
    temp_correct, temp_mistakes, train_pct = evaluate(train_x, train_y, train_x.dot(log_reg_info_away['w']), 'train: ', do_print=False)
    log_reg_info_away['training_accuracy'] = train_pct

    is_svmlight = False
    test_x, test_y = get_xy(data_from_file(test_filepath, is_svmlight=is_svmlight), features_to_drop=away_features_to_drop, is_svmlight=is_svmlight, y_col=away_y_col)
    away_test_y = test_y
    test_x = add_any_missing_features_cols(test_x, num_features)
    test_x = append_bias(test_x)
    temp_correct, temp_mistakes, test_pct = evaluate(test_x, test_y, test_x.dot(log_reg_info_away['w']), 'test: ', do_print=False)
    log_reg_info_away['test_accuracy'] = test_pct

    def win_tie_loss_pct(home_w, away_w, x, home_y, away_y):
        home_preds = x.dot(home_w)
        away_preds = x.dot(away_w)
        correct, mistakes = 0.0, 0.0
        for i in range(len(home_preds)):
            if home_preds[i] * home_y[i] > 0 and away_preds[i] * away_y[i] > 0:
                correct += 1
            else:
                mistakes += 1
        return (correct / float(correct + mistakes)) * 100

    #pdb.set_trace()
    log_reg_info_home['win_tie_loss_accuracy'] = win_tie_loss_pct(log_reg_info_home['w'], log_reg_info_away['w'], test_x, home_test_y, away_test_y)
    log_reg_info_away['win_tie_loss_accuracy'] = log_reg_info_home['win_tie_loss_accuracy']

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
    #         print('log reg', learning_rate, 'of', log_reg_learning_rates, tradeoff, 'of', log_reg_tradeoffs)
    #         w = np.random.rand(num_features + 1, 1) * initial_weight_range - (initial_weight_range / 2.0)
    #         for epoch in range(epochs):
    #             correct, mistakes = 0, 0
    #             for x, y, eval_x, eval_y in cross_val(folds, cv_filepath, num_features=num_features, is_svmlight=False, y_col=y_col, features_to_drop=features_to_drop):
    #                 x, y = permute_examples(x, y)
    #                 w = log_reg(x, y, w, rate=learning_rate, tradeoff=tradeoff)
    #                 temp_correct, temp_mistakes, _ = evaluate(eval_x, eval_y, eval_x.dot(w), 'EPOCH: '+str(epoch + 1), do_print=False)
    #                 correct += temp_correct
    #                 mistakes += temp_mistakes
    #         pct = print_eval_get_pct(correct, mistakes, do_print=False)
    #         if pct > log_reg_info['cv_accuracy']:
    #             log_reg_info['cv_accuracy'] = pct
    #             log_reg_info['learning_rate'] = learning_rate
    #             log_reg_info['tradeoff'] = tradeoff
    #             log_reg_info['w'] = w
    
    # # train set testing
    # train_x, train_y = get_xy(data_from_file(train_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight, features_to_drop=features_to_drop, y_col=y_col)
    # train_x = add_any_missing_features_cols(train_x, num_features)
    # train_x = append_bias(train_x)
    # temp_correct, temp_mistakes, train_pct = evaluate(train_x, train_y, train_x.dot(log_reg_info['w']), 'train: ', do_print=False)
    # log_reg_info['training_accuracy'] = train_pct

    # test_x, test_y = get_xy(data_from_file(test_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight, features_to_drop=features_to_drop, y_col=y_col)
    # test_x = add_any_missing_features_cols(test_x, num_features)
    # test_x = append_bias(test_x)
    # temp_correct, temp_mistakes, test_pct = evaluate(test_x, test_y, test_x.dot(log_reg_info['w']), 'test: ', do_print=False)
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
    #     for x, y, eval_x, eval_y in cross_val(folds, cv_filepath, num_features=num_features, is_svmlight=False, y_col=y_col, features_to_drop=features_to_drop):
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
    # train_x, train_y = get_xy(data_from_file(train_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight, features_to_drop=features_to_drop, y_col=y_col)
    # train_x = add_any_missing_features_cols(train_x, num_features)
    # train_x = append_bias(train_x)
    # train_preds = []
    # for i in range(len(train_x)):
    #     train_preds.append(bayes_info['b'].predict(train_x, i))
    # temp_correct, temp_mistakes, train_pct = evaluate(train_x, train_y, train_preds, 'EPOCH: ', do_print=False)
    # bayes_info['training_accuracy'] = train_pct

    # is_svmlight = True
    # train_x, train_y = get_xy(data_from_file(test_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight, features_to_drop=features_to_drop, y_col=y_col)
    # test_x = add_any_missing_features_cols(test_x, num_features)
    # test_x = append_bias(test_x)
    # test_preds = []
    # for i in range(len(test_x)):
    #     test_preds.append(bayes_info['b'].predict(test_x, i))
    # temp_correct, temp_mistakes, test_pct = evaluate(test_x, test_y, test_preds, 'test: ', do_print=False)
    # bayes_info['test_accuracy'] = test_pct

    print('SVM Home')
    print('best hyperparameters:')
    print('- learning rate', svm_info_home['learning_rate'])
    print('- regularization', svm_info_home['regularization'])
    print('cv accuracy', svm_info_home['cv_accuracy'])
    print('training accuracy', svm_info_home['training_accuracy'])
    print('test accuracy', svm_info_home['test_accuracy'])
    print()
    print('SVM Away')
    print('best hyperparameters:')
    print('- learning rate', svm_info_away['learning_rate'])
    print('- regularization', svm_info_away['regularization'])
    print('cv accuracy', svm_info_away['cv_accuracy'])
    print('training accuracy', svm_info_away['training_accuracy'])
    print('test accuracy', svm_info_away['test_accuracy'])
    print('WIN/TIE/LOSS PRED %', svm_info_away['win_tie_loss_accuracy'])
    print()

    print('log reg Home')
    print('best hyperparameters:')
    print('- learning rate', log_reg_info_home['learning_rate'])
    print('- tradeoff', log_reg_info_home['tradeoff'])
    print('cv accuracy', log_reg_info_home['cv_accuracy'])
    print('training accuracy', log_reg_info_home['training_accuracy'])
    print('test accuracy', log_reg_info_home['test_accuracy'])
    print()
    print('log reg Away')
    print('best hyperparameters:')
    print('- learning rate', log_reg_info_away['learning_rate'])
    print('- tradeoff', log_reg_info_away['tradeoff'])
    print('cv accuracy', log_reg_info_away['cv_accuracy'])
    print('training accuracy', log_reg_info_away['training_accuracy'])
    print('test accuracy', log_reg_info_away['test_accuracy'])
    print('WIN/TIE/LOSS PRED %', log_reg_info_away['win_tie_loss_accuracy'])
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
