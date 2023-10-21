
import os


uppercase_lm_path = 'default_test_model/3-gram.pruned.1e-7.arpa'


lm_path = 'default_test_model/lowercase_3-gram.pruned.1e-7.arpa'
if not os.path.exists(lm_path):
    with open(uppercase_lm_path, 'r') as f_upper:
        with open(lm_path, 'w') as f_lower:
            for line in f_upper:
                f_lower.write(line.lower())
print('Converted language model file to lowercase.')