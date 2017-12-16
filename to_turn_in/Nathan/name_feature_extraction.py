run_tests = False

def t(i, o):
    if run_tests:
        if i != o:
            raise Exception('Failed Test!')

def parse_data_line(line):
    return line.split(' ')[0], line[2:].strip()

f = parse_data_line
i, o = '+ David Gelernter', ('+', 'David Gelernter')
t(f(i), o)


def read_in_data(path):
    data = []
    with open(path, 'r') as r:
        for line in r:
            data.append(parse_data_line(line))
    return data


# In[4]:

# extract features
def first_last_middle(name):
    names = name.split(' ')
    return names[0], names[1:-1], names[-1]

f = first_last_middle
i, o = 'C. A. R. Hoare', ('C.', ['A.', 'R.'], 'Hoare')
t(f(i), o)
i, o = 'James H. Wilkinson', ('James', ['H.'], 'Wilkinson')
t(f(i), o)
i, o = 'FranÃ§ois Vernadat', ('FranÃ§ois', [], 'Vernadat')
t(f(i), o)


# In[5]:

def first_name_longer_than_last(first, last):
    return len(first) > len(last)

f = first_name_longer_than_last
i1, i2, o = 'Moshe', 'Vardi', False
t(f(i1, i2), o)
i1, i2, o = 'Vladimir', 'Vapnik', True
t(f(i1, i2), o)
i1, i2, o = 'Arif', 'Zaman', False
t(f(i1, i2), o)


# In[6]:

def has_middle_name(middle):
    return len(middle) > 0

f = has_middle_name
i, o = ['L.', 'Steele,'], True
t(f(i), o)
i, o = [], False
t(f(i), o)
i, o = ['B.'], True
t(f(i), o)


# In[7]:

def same_first_letter(first, last):
    return first.lower()[0] == last.lower()[0]

f = same_first_letter
i1, i2, o = 'Michael', 'Scott', False
t(f(i1, i2), o)
i1, i2, o = 'Paritosh', 'Pandya', True
t(f(i1, i2), o)
i1, i2, o = 'Paritosh', 'pandya', True
t(f(i1, i2), o)
i1, i2, o = 'paritosh', 'Pandya', True
t(f(i1, i2), o)


# In[8]:

def first_before_last(first, last):
    return sorted(first) < sorted(last)

f = first_before_last
i1, i2, o = 'Carroll', 'Morgan', True
t(f(i1, i2), o)
i1, i2, o = 'Tom', 'Mitchell', False
t(f(i1, i2), o)
i1, i2, o = 'Levin', 'Levin', False
t(f(i1, i2), o)
i1, i2, o = 'Yann', 'leCun', False
t(f(i1, i2), o)
i1, i2, o = 'yann', 'LeCun', False
t(f(i1, i2), o)
i1, i2, o = 'YaNa', 'Yann', True
t(f(i1, i2), o)


# In[9]:

def second_letter_vowel(name, vowels = ['a', 'e', 'i', 'o', 'u']):
    if len(name) < 2:
        return False
    return name.lower()[1] in vowels

f = second_letter_vowel
i, o = 'Manny', True
t(f(i), o)
i, o = 'Andrew', False
t(f(i), o)
i, o = 'BÃ¶rje', False
t(f(i), o)
i, o = '', False
t(f(i), o)
i, o = '.', False
t(f(i), o)


# In[10]:

def num_chars_even(name):
    return not bool(len(name) % 2)

f = num_chars_even
i, o = 'Kolmogorov', True
t(f(i), o)
i, o = 'Etzioni', False
t(f(i), o)
i, o = '', True
t(f(i), o)


# In[21]:

def length_less_than_eight(name):
    return len(name) < 8


# In[25]:

def length_more_than_nine(name):
    return len(name) > 9


# In[12]:

def parse_label(label):
    if label == '+':
        return True
    elif label == '-':
        return False
    else:
        raise Exception('Unknown Label: {}'.format(label))

f = parse_label
i, o = '+', True
t(f(i), o)
i, o = '-', False
t(f(i), o)


# In[26]:

class NameExample:
    def __init__(self, _id, full_name, label):
        self.id = _id
        self.full_name = full_name
        first, middle, last = first_last_middle(full_name)
        self.first = first
        self.middle = middle
        self.last = last
        self.first_longer_last = first_name_longer_than_last(first, last)
        self.has_middle_name = has_middle_name(middle)
        self.same_first_char = same_first_letter(first, last)
        self.first_before_last = first_before_last(first, last)
        self.has_second_char_vowel = second_letter_vowel(first)
        self.last_num_chars_even = num_chars_even(last)
        self.length_less_than_eight = length_less_than_eight(full_name)
        self.length_more_than_nine = length_more_than_nine(full_name)
        self.label = parse_label(label)

# Just test for exception
f = NameExample
i1, i2, i3 = 7, 'Brendan Eich', '-'
if run_tests:
    o = f(i1, i2, i3)
    t(o.id, 7)
    t(o.full_name, 'Brendan Eich')
    t(o.first, 'Brendan')
    t(o.middle, [])
    t(o.last, 'Eich')
    t(o.first_longer_last, True)
    t(o.has_middle_name, False)
    t(o.same_first_char, False)
    t(o.has_second_char_vowel, False)
    t(o.last_num_chars_even, True)
    t(o.label, False)

def extract_id_to_name_example(data):
    _id = 0
    id_to_name_example = {}
    for row in data:
        label = row[0]
        name = row[1]
        name_example = NameExample(_id, name, label)
        id_to_name_example[_id] = name_example
        _id += 1
    return id_to_name_example


# In[24]:

def featurized_data(path):
    data = read_in_data(path)
    return extract_id_to_name_example(data)


# In[15]:
_ = featurized_data('Dataset/training.data')


# In[ ]:




# In[ ]:



