import os
import json
import pickle as pkl
import re

def preprocess_stories(home_dir, pickle_dir, story_data, story_keys):
    
    os.chdir(pickle_dir) # change to pkl directory

    # preprocess, only grab stories that are not missing pickle files
    missing_stories = []

    for key in story_keys: # grab the story_ids
        for image in story_data[key]['images']: # image list of each story_id
        
            if image + '.pkl' not in os.listdir(pickle_dir): # check which stories have missing pkl files
                if key not in missing_stories:
                    missing_stories.append(key)
                    
    print(f'Number of story ids with missing images: {len(missing_stories)}')
            
    # remove stories with missing images
    for story in missing_stories:
        story_keys.remove(story)

    print(f"Finished! New number of story ids: {len(story_keys)}")
    
    # go back to home dir
    os.chdir(home_dir)
    
    
    return story_keys


def grab_features(home_dir, pickle_dir, story_data, story_keys):
    
    os.chdir(pickle_dir) # change to pkl directory

    features = []
    img_names = []
    
    new_len = 0 # used for printing

    for key in story_keys: # grab the story_ids
        for image in story_data[key]['images']: # image list of each story_id
        
            if image + '.pkl' in os.listdir(pickle_dir): # check if in pickle_dir
                
                pkl_file = open(image + '.pkl', 'rb') # open pickle file
                one_feature = pkl.load(pkl_file) # grab feature

                features.append(one_feature) # store features

                img_names.append(int(image)) # stores the id of the image
                
                # loading print change this to %1000 == 0
                if len(features) == 1000:
                    print(f"Extracted {len(features)} features")
                    break
        if len(features) == 1000:
                    print(f"Extracted {len(features)} features")
                    break        

    print(f'Finished! Length of features: {len(features)}')
    
    # go back to home dir
    os.chdir(home_dir)

    return features, img_names


def vect_sentences(story_data, story_keys, vocab):
    
    # vectorize

    vector_sentences = []

    for key in story_keys: # grab each story id
        for sentence in story_data[key]['sentences']: # grab sentence from sentence list

            res = re.findall(r'\w+', sentence) # grabs only the words of the sentence

            v_sentence = [] # vectorize the sentence
            for word in res:
                word = word.lower()
                v_word = vocab[word]
                v_sentence.append(v_word)
            vector_sentences.append(v_sentence) # store the set of 5 sentences

            if len(vector_sentences) == 10000:
                break
        if len(vector_sentences) == 10000:
                break

    print(f'Finished! Length of sentences: {len(vector_sentences)}') 

    return vector_sentences

# def ids_to_words: