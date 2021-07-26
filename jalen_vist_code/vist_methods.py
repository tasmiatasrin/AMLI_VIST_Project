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

def padding(v_sentence):

    if len(v_sentence) < 18:
        while len(v_sentence) != 18:
            v_sentence.append(-1)

    else:
        while len(v_sentence) != 18:
            v_sentence = v_sentence[:-1]

    # print(len(v_sentence))
    return v_sentence


def grab_features(home_dir, pickle_dir, story_data, story_keys, vocab):
    
    os.chdir(pickle_dir) # change to pkl directory

    features = []
    img_names = []
    all_sentences = []

    new_len = 0 # used for printing

    for key in story_keys: # grab the story_ids
        for i in range(5):
            image = story_data[key]['images'][i]

            sentence = story_data[key]['sentences'][i]
        
            if image + '.pkl' in os.listdir(pickle_dir): # check if in pickle_dir
                
                pkl_file = open(image + '.pkl', 'rb') # open pickle file
                one_feature = pkl.load(pkl_file) # grab feature

                features.append(one_feature) # store features

                img_names.append(int(image)) # stores the id of the image

                all_sentences.append(sentence)

        if len(all_sentences)%100 == 0:
            print(len(all_sentences))
            print(all_sentences[50])
            print(img_names[50])
            break

                # res = re.findall(r'\w+', sentence) # grabs only the words of the sentence

                # v_sentence = [] # vectorize the sentence
                # for word in res:
                #     word = word.lower()
                #     v_word = vocab[word]
                #     v_sentence.append(v_word)


                # # add the padding
                #     if len(v_sentence) != 18:
                #         v_sentence = padding(v_sentence)

                #         vector_sentences.append(v_sentence) # store the set of 5 sentences
                
                # loading print change this to %1000 == 0
        #         if len(features) == 100:
        #             print(f"Extracted {len(features)} features")
        #             break
        # if len(features) == 100:
        #             print(f"Extracted {len(features)} features")
        #             break        

    print(f'Finished! Length of features: {len(features)}')
    
    # go back to home dir
    os.chdir(home_dir)

    return features, img_names, all_sentences



def vect_sentences(sentences, vocab):
    
    # vectorize

    vector_sentences = []
    len_sent = []

    for sentence in sentences: # grab each sentence

        res = re.findall(r'\w+', sentence) # grabs only the words of the sentence

        v_sentence = [] # vectorize the sentence
        for word in res:
            word = word.lower()
            v_word = vocab[word]
            v_sentence.append(v_word)


            # add the padding
        if len(v_sentence) != 18:
            v_sentence = padding(v_sentence)

        vector_sentences.append(v_sentence) # store the set of 5 sentences
            
            # len_sent.append(len(v_sentence)) # store the length of vectorized sentences

        #     if len(vector_sentences) == 100:
        #         break
        # if len(vector_sentences) == 100:
        #     break

    # use this to test the length, it should be 18
    # percentile = np.percentile(len_sent, 95)
    # print(percentile)    

    print(f'Finished! Length of sentences: {len(vector_sentences)}') 

    return vector_sentences