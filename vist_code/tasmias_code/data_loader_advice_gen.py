import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab_v2 import Vocabulary
from frogger_dataset import FroggerDataset
import cv2

max_word_num = 17
class FroggerDataLoader(data.Dataset):
    def __init__(self, vocab, rationalizations, images, cur_image_dir):
        
        #print("rationalizations: ",rationalizations )
        self.vocab = vocab
        self.rationalizations = rationalizations
        #print("self.rationalizations: ",len(self.rationalizations))
        self.images = images
        #print("self.images: ",len(self.images))
        self.image_dir = cur_image_dir
        self.img_height = 100
        self.img_width = 100
        
       
    
    def __getitem__(self, index):
		
        text_matrix = self.rationalizations[index]
        length_advice = len(text_matrix)
        image_name = self.images[index]

        img_path = os.path.join(self.image_dir, image_name)
        img_arr = cv2.imread(img_path)
        resized_img = cv2.resize(img_arr, (self.img_height,self.img_width))
        
        x = np.reshape(resized_img, (3,self.img_height,self.img_width))
        target_text = torch.Tensor(text_matrix)
        image_arr = torch.Tensor(x)

        return image_name, image_arr, target_text


    def __len__(self):
        return len(self.rationalizations)

def collate_fn(data):
	"""Creates mini-batch tensors from the list of tuples (image, caption).	
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
	data_length = min(max(lengths), max_word_num)
	#targets = torch.zeros(len(captions), max(lengths)).long()
	#targets = torch.zeros(len(captions), max(lengths)).long()
	targets = torch.zeros(len(captions),data_length).long()
	for i, cap in enumerate(captions):
		end = min(lengths[i], data_length)
		targets[i, :end] = cap[:end] 
	return names, images, targets		
	#return images, targets, lengths


def get_loader( vocab, rationalizations, images, cur_image_dir, batch_size,transform, shuffle, num_workers):
    frogger = FroggerDataLoader(vocab, rationalizations, images, cur_image_dir)

    data_loader = torch.utils.data.DataLoader(dataset=frogger, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader
