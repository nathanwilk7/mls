import pdb

import numpy as np

from cross_validation import cross_val, data_from_file, get_xy, append_bias, add_any_missing_features_cols
from evaluation import evaluate, print_eval_get_pct
from perceptron_alg import perceptron

epochs = 10
folds = 5
cv_filepath = 'data/past_matches_results_train_cv_{cv}.csv' #'Dataset/CVSplits/training0{cv}.data'
initial_weight_range=0.02
#features_to_drop = ['away_result', 'score_home', 'score_away', 'outcome', 'home_team_name', 'away_team_name', 'competition_id', 'season_id']
features_to_drop = ['away_result', 'score_home', 'score_away', 'outcome', 'home_team_name', 'away_team_name', 'competition_id', 'season_id', 'home_half_score_home', 'home_half_score_away', 'away_half_score_home', 'away_half_score_away']
# features_to_drop = [#'away_result',
#     'Unnamed: 0', 'Unnamed: 0.1',
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
y_col = 'home_result'
num_features = 392
learning_rates = [1, 0.1, 0.01]
margins = [1, 0.1, 0.01]
dev_filepath = 'data/past_matches_results_train.csv'
test_filepath = 'data/past_matches_results_test.csv'
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
    is_svmlight = False
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
                import pandas as pd
                for x, y, eval_x, eval_y in cross_val(folds, cv_filepath, num_features=num_features, is_svmlight=False, y_col=y_col, features_to_drop=features_to_drop):
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
                    temp_correct, temp_mistakes, _ = evaluate(eval_x, eval_y, eval_x.dot(w), 'EPOCH: '+str(epoch + 1), do_print=False)
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
    dev_x, dev_y = get_xy(data_from_file(dev_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight, features_to_drop=features_to_drop, y_col=y_col)
    dev_x = add_any_missing_features_cols(dev_x, num_features)
    dev_x = append_bias(dev_x)
    best_alg['dev_plot_y'] = []
    for epoch in range(dev_epochs):
        correct, mistakes = 0, 0
        if dynamic_learning_rate_epochs:
            learning_rate = initial_learning_rate / (1 + epoch)
        for x, y, _, _ in cross_val(folds, cv_filepath, num_features=num_features, is_svmlight=False, y_col=y_col, features_to_drop=features_to_drop):
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
            temp_correct, temp_mistakes, _ = evaluate(dev_x, dev_y, dev_x.dot(w), 'dev epoch: '+str(epoch + 1), do_print=False)
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

    test_x, test_y = get_xy(data_from_file(test_filepath, is_svmlight=is_svmlight), is_svmlight=is_svmlight, features_to_drop=features_to_drop, y_col=y_col)
    test_x = add_any_missing_features_cols(test_x, num_features)
    test_x = append_bias(test_x)
    temp_correct, temp_mistakes, test_pct = evaluate(test_x, test_y, test_x.dot(best_alg['w']), 'test: ')
    best_alg['test_pct'] = test_pct
    # to make printing cleaner
    best_alg['w'] = None
    print(best_alg)
    print('training updates:', training_updates)

perceptron_wrapper('simple perceptron', learning_rates=learning_rates)
perceptron_wrapper('dynamic learning rate perceptron (on epochs)', learning_rates=learning_rates, dynamic_learning_rate_epochs=True)
perceptron_wrapper('dynamic learning rate perceptron (on examples)', learning_rates=learning_rates, dynamic_learning_rate_examples=True)
perceptron_wrapper('margin perceptron', learning_rates=learning_rates, margins=margins, dynamic_learning_rate_epochs=True)
perceptron_wrapper('averaged perceptron', learning_rates=learning_rates, averaged=True)
