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
        
        return img_name, x, y
        
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption)
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            image: torch tensor of shape (3, 256, 256).
            caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[2]), reverse=True)
    names, images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    data_length = min(max(lengths), 18)
    #targets = torch.zeros(len(captions), max(lengths)).long()
    #targets = torch.zeros(len(captions), max(lengths)).long()
    targets = torch.zeros(len(captions),data_length).long()
    for i, cap in enumerate(captions):
        end = min(lengths[i], data_length)
        targets[i, :end] = cap[:end] 
    return names, images, targets		
    #return images, targets, lengths


    

def get_loader(vocab, image_features, image_names, v_sent, transform, batch_size, shuffle, num_workers): # maybe add transform, pkl_dir, vocab
    vist = VistDataLoader(vocab, v_sent, image_features, image_names)
    
    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=batch_size, num_workers=num_workers,collate_fn=collate_fn) # default collate_fn
    return data_loader

# advice_gen_full
