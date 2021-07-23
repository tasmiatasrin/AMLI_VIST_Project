import methods.py

# grab story dictionary

import os
import json
import pickle as pkl
import re


! pwd

home_dir = '/Users/javi/Desktop/AMLI/Capstone/2_vocabulary_and_data_loader'

os.chdir(home_dir)

! pwd

pickle_dir = '/Users/javi/Desktop/AMLI/Capstone/1_story_dictionary_and_pickles/pickles/'

# open JSON file
d = open('tsplit1_dictionary.json')
v = open('6_complete_vocabulary.json') # vocabulary

# returns JSON object as a dictionary

story_data = json.load(d)
vocab = json.load(v)

# close file
d.close
v.close

# grab the keys
story_keys = [key for key in story_data.keys()]
print(f'Number of story ids: {len(story_keys)}')