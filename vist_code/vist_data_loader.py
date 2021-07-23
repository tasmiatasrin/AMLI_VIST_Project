import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image

class VistDataLoader(data.Dataset):
    
    def __init__(self, vocab, v_sent, img_features, img_names): #story_keys, vocab, batch_size)# home_dir, pickle_dir):
        # self.keys = story_keys
        # self.vocab = vocab
        # self.batch_size = 8
        self.img_names = img_names
        self.img_features = img_features
        self.v_sent = v_sent
        self.vocab = vocab

        # self.home_dir = home_dir
        # self.pickle_dir = pickle_dir
        
        
    def __len__(self):
        # if len(self.img_features) != len(self.v_sent):
        #     print("Number of images should match sentences")
        #     return
        return len(self.img_features)
        
        
    def __getitem__(self, index):

        img_name = self.img_names[index]

        img_features = self.img_features[index]
        
        vect_sent = self.v_sent[index]

        x = torch.Tensor(img_features)
        y = torch.Tensor(vect_sent) # add padding if this throws an error
        
        # print(y, len(y), type(y)) #img_name, len(img_name), x, len(x), 

        # print(len(img_name), len(x), len(y))

        return img_name, x, y
        

        

def get_loader(vocab, image_features, image_names, v_sent, transform, batch_size, shuffle, num_workers): # maybe add transform, pkl_dir, vocab
    vist = VistDataLoader(vocab, v_sent, image_features, image_names)

    # print(f"hellooo {len(vist.v_sent)}")
    print(f' I think this tis the length of the train/val vectorized sentences: {vist.v_sent[0]}')
    print(f'How do I just grab a "batch_size" of sentences?')
    
    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=batch_size, num_workers=num_workers) # default collate_fn
    return data_loader

# advice_gen_full
