import os
import json
import pickle as pkl
import re
import torch


class DataLoader(data.Dataset):
    
    def __init__(self, story_dict, story_keys, vocab, batch_size, home_dir, pickle_dir):
        self.dict = story_dict
        self.keys = story_keys
        self.vocab = vocab
        self.batch_size = 8
        
        # do I need this?
        self.home_dir = home_dir
        self.pickle_dir = pickle_dir
        
        self.images = grab_features(home_dir, pickle_dir, story_data, story_keys)
        self.sentences = vect_sentences(story_data, story_keys, vocab)
        
        
    def __len__(self):
        if len(self.images) != len(self.sentences):
            print("Number of images should match sentences")
            return
        return len(images)
        
        
    def __getitem__(self, index):

        
        image_features = self.images[index]
        
        vect_sent = self.sentences[index]

        x = torch.Tensor(image_features)
        y = torch.Tensor(vect_sent) # add padding if this throws an error
        
        return x, y
        

        

def get_loader( vocab, sentences, images, pickle_dir, home_dir, batch_size, transform, num_workers=0):
    vist = DataLoader(story_data, story_keys, vocab, batch_size, home_dir, pickle_dir)
    
    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=8, num_workers=num_workers)
    return data_loader

# advice_gen_full
