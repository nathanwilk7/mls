
# coding: utf-8

# In[146]:

import unittest as ut

import name_feature_extraction as feat_ext

run_tests = True


# In[147]:

class Node:
    def __init__(self, 
                 attr=None, 
                 value=None,
                 output=None):
        self.attr = attr
        self.value = value
        self.output = output
        self.children = []


# In[148]:

from math import log

def entropy(examples, label):
    trues, falses = 0.0, 0.0
    for _id, example in examples.items():
        if example.label:
            trues += 1
        else:
            falses += 1
    if trues == 0 or falses == 0:
        return 0
    total = trues + falses
    proportion_true = trues / total
    proportion_false = falses / total
    return (-(proportion_true * log(proportion_true, 2) +
              proportion_false * log(proportion_false, 2)))


# In[149]:

def get_examples_with_attr(examples, attr):
    res = {}
    for _id, example in examples.items():
        if getattr(example, attr):
            res[_id] = examples[_id]
    return res


# In[150]:

def get_examples_without_attr(examples, attr):
    res = {}
    for _id, example in examples.items():
        if not getattr(example, attr):
            res[_id] = examples[_id]
    return res


# In[151]:

def get_examples_with_attr_and_value(examples, attr, value):
    res = {}
    for _id, example in examples.items():
        if getattr(example, attr) == value:
            res[_id] = examples[_id]
    return res


# In[152]:

def choose_attribute_max_info_gain(examples, attrs, label, info_val_func=entropy):
    min_so_far = 1
    min_attr = None
    for attr in attrs:
        examples_with_attr = get_examples_with_attr(examples, attr)
        info_val = info_val_func(examples_with_attr, label)
        examples_without_attr = get_examples_without_attr(examples, attr)
        info_val_without = info_val_func(examples_without_attr, label)
        total_examples = len(examples)
        info_gain_ish = ((float(len(examples_with_attr)) / float(total_examples)) * info_val
                         + float(len(examples_without_attr) / float(total_examples)) * info_val_without)
        if info_gain_ish < min_so_far:
            min_so_far = info_gain_ish
            min_attr = attr
    return min_attr


# In[153]:

def all_examples_have_same_label(examples, label):
    last_label = None
    for _id, example in examples.items():
        if last_label is not None and example.label != last_label:
            return False, last_label
        last_label = example.label
    return True, last_label


# In[154]:

def possible_values_of_attr(examples, attr):
    possible_values = set()
    for _id, example in examples.items():
        possible_values.add(getattr(example, attr))
    return possible_values


# In[155]:

def most_common_label(examples, label):
    trues, falses = 0, 0
    for _id, example in examples.items():
        if example.label:
            trues += 1
        else:
            falses += 1
    return trues >= falses


# In[156]:

# attr in parent node, condition in child node
def id3(node, examples, attributes, label, attribute_chooser):
    all_same_label, last_label = all_examples_have_same_label(examples, label)
    if all_same_label and last_label is not None:
        node.output = last_label
        return node
    else:
        best_attr = choose_attribute_max_info_gain(examples, attributes, label, info_val_func=entropy)
        node.attr = best_attr
        if best_attr is None:
#             import pdb
#             pdb.set_trace()
            common_label = most_common_label(examples, label)
            node.output = common_label
            return node
        attributes.remove(best_attr)
#         possible_vals = possible_values_of_attr(examples, best_attr)
        for possible_val in [False, True]:
            examples_with_val = get_examples_with_attr_and_value(examples, best_attr, possible_val)
            if len(examples_with_val) == 0:
                common_val = most_common_label(examples, label)
                child_node = Node(value=possible_val, output=common_val)
                node.children.append(child_node)
            else:
                child_node = id3(Node(value=possible_val), examples_with_val, attributes, label, attribute_chooser)
                node.children.append(child_node)
        attributes.add(best_attr)
        return node


# In[158]:

attributes = {'first_before_last'} #{'first_longer_last', 'has_middle_name', 'same_first_char', 
             #'first_before_last', 'has_second_char_vowel', 'last_num_chars_even', 
             # 'length_less_than_eight', 'length_more_than_nine'}

training_path = 'Updated_Dataset/updated_train.txt'
training_id_to_name_examples = feat_ext.featurized_data(training_path)

root = id3(Node(), training_id_to_name_examples, attributes, 'label', choose_attribute_max_info_gain)


# In[120]:

def predict_example(node, example):
    if node.output is not None:
        return node.output
    for child in node.children:
        if getattr(example, node.attr) == child.value:
            return predict_example(child, example)
    else:
        print('could not predict')
        return node.output


# In[121]:

def eval_examples(node, examples):
    correct, wrong = 0, 0
    for _id, example in examples.items():
        predicted = predict_example(node, example)
        if predicted == example.label:
            correct += 1
        else:
            wrong += 1
    return correct, wrong


# In[122]:

train_correct, train_wrong = eval_examples(root, training_id_to_name_examples)
print('Training Score')
print(train_correct, train_wrong)
print(float(train_correct) / (train_correct + train_wrong))


# In[124]:

test_path = 'Updated_Dataset/updated_test.txt'
test_id_to_name_examples = feat_ext.featurized_data(test_path)
test_correct, test_wrong = eval_examples(root, test_id_to_name_examples)
print('Test Score')
print(test_correct, test_wrong)
print(float(test_correct) / (test_correct + test_wrong))


# In[33]:

# class TestDecisionTree(ut.TestCase):
#     def test_node(self):
#         attr = 'first_longer_last'
#         value = True
#         children = [Node(), Node()]
#         node = Node(attr=attr, 
#                     value=value,
#                     children=children)
#         self.assertEqual(node.attr, attr)
#         self.assertEqual(node.value, value)
#         self.assertEqual(node.children, children)
        
# #     def test_id3_simple(self):
# #         _id = 0
# #         example0 = feat_ext.NameExample(_id, 'Tim Finin', '+')
# #         id_to_name_examples = {}
# #         id_to_name_examples[_id] = example0
# #         attributes = {'first_longer_last',
# #                       'has_middle_name',
# #                       'same_first_char',
# #                       'first_before_last',
# #                       'has_second_char_vowel',
# #                       'last_num_chars_even'
# #                      }
# #         label = 'label'
# #         trained_model = id3(id_to_name_examples, attributes, label)
# #         output_model = Node(output=True)
# #         self.assertEqual(trained_model.output, output_model.output)
        
# #     def test_id3(self):
# #         _id = 0
# #         example0 = feat_ext.NameExample(_id, 'Tim Finin', '+')
# #         id_to_name_examples = {}
# #         id_to_name_examples[_id] = example0
# #         attributes = {'first_longer_last',
# #                       'has_middle_name',
# #                       'same_first_char',
# #                       'first_before_last',
# #                       'has_second_char_vowel',
# #                       'last_num_chars_even'
# #                      }
# #         label = 'label'
# #         trained_model = id3(id_to_name_examples, attributes, label)
# #         output_model = Node(output=True)
# #         self.assertEqual(trained_model.output, output_model.output)

# if run_tests:
#     suite = ut.TestLoader().loadTestsFromTestCase(TestDecisionTree)
#     ut.TextTestRunner(verbosity=1).run(suite)

