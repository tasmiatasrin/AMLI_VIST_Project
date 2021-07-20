import os
import json
import pickle as pkl
import re
import torch


class DataLoader(data.Dataset):
    
    def __init__(self, story_keys, vocab, batch_size, home_dir, pickle_dir):
        self.keys = story_keys
        self.vocab = vocab
        self.batch_size = 8

        self.image_features
        self.vect_sentences

        self.home_dir = home_dir
        self.pickle_dir = pickle_dir
        
        
    def __len__(self):
        if len(self.images) != len(self.sentences):
            print("Number of images should match sentences")
            return
        return len(images)
        
        
    def __getitem__(self, index):

        
        img_features = self.image_features[index]
        
        vect_sent = self.vect_sentences[index]

        x = torch.Tensor(img_features)
        y = torch.Tensor(vect_sent) # add padding if this throws an error
        
        return x, y
        

        

def get_loader(vocab, sentences, images, pickle_dir, home_dir, batch_size, transform, num_workers=0):
    vist_dataset = DataLoader(story_keys, vocab, batch_size, home_dir, pickle_dir)
    
    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=8, num_workers=num_workers)
    return data_loader

# advice_gen_full
