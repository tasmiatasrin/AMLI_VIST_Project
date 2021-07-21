import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image

class VistDataLoader(data.Dataset):
    
    def __init__(self, vocab, v_sent, img_features): #story_keys, vocab, batch_size)# home_dir, pickle_dir):
        # self.keys = story_keys
        # self.vocab = vocab
        # self.batch_size = 8

        self.img_features = img_features
        self.v_sent = v_sent
        self.vocab = vocab

        # self.home_dir = home_dir
        # self.pickle_dir = pickle_dir
        
        
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
        

        

def get_loader(vocab, img_features, v_sent, transform, batch_size, shuffle, num_workers): # maybe add transform, pkl_dir, vocab
    vist = VistDataLoader(vocab, v_sent, img_features)
    
    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=batch_size, num_workers=num_workers) # default collate_fn
    return data_loader

# advice_gen_full
