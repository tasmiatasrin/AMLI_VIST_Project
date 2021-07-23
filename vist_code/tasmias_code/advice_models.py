import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import tensorflow as tf
import json
import copy
import numpy as np
# from xlrd import open_workbook
from collections import Counter
from torchvision import transforms
#from data_loader import FroggerDataLoader, get_loader, create_rationalization_matrix
# from data_loader_advice_driven_train import FroggerDataLoader, get_loader, create_rationalization_matrix
# import nltk
import re
#from build_vocab_v2 import Vocabulary
from math import floor
# import cv2
# from sklearn.utils import shuffle
from PIL import Image
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout2d(p=0.2)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        #print("images shape: ", images.shape)
        features = self.resnet(images)
        #print("features shape: ", features.shape)
        features = features.view(features.size(0), -1)
        #print("features shape after view: ", features.shape)
        features = self.dropout(features)
        #print("features shape after dropout: ", features.shape)
        features = self.embed(features)
        # print("features shape after embed: ", features.shape)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, word_embed_size, image_emb_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, word_embed_size)
        
        self.lstm = nn.LSTM(input_size = word_embed_size+image_emb_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        # self.hid_linear1 = nn.Linear(image_emb_size, hidden_size)
        # self.hid_linear2 = nn.Linear(image_emb_size, hidden_size)
        self.dropout = nn.Dropout2d(p=0.5)
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.word_embed_size = word_embed_size
        self.vocab_size = vocab_size
        self.image_emb_size = image_emb_size
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        # self.hid_linear1.weight.data.normal_(0.0, 0.02)
        # self.hid_linear1.bias.data.fill_(0)
        # self.hid_linear2.weight.data.normal_(0.0, 0.02)
        # self.hid_linear2.bias.data.fill_(0)
    
    def init_word_embedding(self, weight_init):
        assert weight_init.shape == (self.vocab_size, self.word_embed_size)
        self.embedding_layer.weight.data[:self.vocab_size] = weight_init

    def init_hidden(self, features):
        batch_size = features.shape[0]
        # _h = self.hid_linear1(features)
        # _c = self.hid_linear2(features)
        # h0 = _h.view(1 * self.n_layers, batch_size, self.hidden_size)
        # c0 = _c.view(1 * self.n_layers, batch_size, self.hidden_size)
        h0 = torch.zeros(1 * self.n_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(1 * self.n_layers, batch_size, self.hidden_size)
        
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
            
        return (h0, c0)

    def forward(self, features, captions):
        
        # captions = captions[:, :-1]
        caption_shape = captions.shape
        embeddings = self.embedding_layer(captions)
        (h,c) = self.init_hidden(features)
        features = features.unsqueeze(1)
        features = features.expand(-1, caption_shape[1], -1)
        lstm_input = torch.cat((features, embeddings), 2) 
        
        # embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        
        
        lstm_outputs, _ = self.lstm(lstm_input,(h,c))
        lstm_outputs = self.dropout(lstm_outputs)
        out = self.linear(lstm_outputs)
        predicted = torch.argmax(out, dim =2) 
        return out

    def inference(self, features):
        batch_size = features.shape[0]
        (hn, cn) = self.init_hidden(features)
        
        sampled_ids = []
        features = features.unsqueeze(1)
        predicted = torch.tensor([1], dtype=torch.long).cuda()
        embed = self.embedding_layer(predicted)
        embed = embed.unsqueeze(1)
        embed = embed.expand(batch_size, -1, -1)
        lstm_input = torch.cat((features, embed), 2)
        for i in range(10):
            lstm_outputs, (hn, cn) = self.lstm(lstm_input, (hn, cn))
            # lstm_outputs = self.dropout(lstm_outputs)
            output = self.linear(lstm_outputs)
            output = output.squeeze(1)
            scores = F.log_softmax(output, dim=1)            
            _predicted = torch.argmax(output, dim=1)
            predicted = scores.max(1)[1]
            sampled_ids.append(predicted)
            embed = self.embedding_layer(predicted).unsqueeze(1)
            lstm_input = torch.cat((features, embed), 2)
        
        return sampled_ids
    
class image_EncoderCNN(nn.Module):
    def __init__(self, image_embedding_size):
        super(image_EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,stride=2)
        #self.max1= nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3,stride=2)
        self.max2= nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(32,32, 3,stride=1)
        #self.max3= nn.MaxPool2d(2,1)
        #self.flatten = torch.flatten()
        
        #self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout2d(p=0.2)
        fc1_size = 1024
        self.fc1 = nn.Linear(3872, fc1_size)
        self.embed = nn.Linear(fc1_size, image_embedding_size)
        #self.init_weights()

    def init_weights(self):
        self.fc1.weight.data.normal_(0.0, 0.02)
        self.fc1.bias.data.fill_(0)

    def forward(self, images):
        #print("images shape: ", images.shape)
        conv1 = self.conv1(images)
        features1 = F.relu(conv1)
        #features1= F.max_pool2d (conv1, 2, stride = 2) 
        conv2 = self.conv2(features1)
        conv2 = F.relu(conv2)
        features2= F.max_pool2d(conv2, 2,stride = 2)
        conv3 = self.conv3(conv2)
        conv3 = F.relu(conv3)
        features3= F.max_pool2d(conv3, 2,stride = 2)
        batch = features3.shape[0]
        #print("features3_shape: ",features3.shape)
        flat_f= torch.reshape(features3,(batch,-1))
        #drop_fc = self.dropout(flat_f)
        #print("flat_f_shape: ",flat_f.shape)
        fc = self.fc1(flat_f)
        features = self.embed(fc)

        return features


class WordEmbedding(nn.Module):
    """Word Embedding
    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, advice_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken+1, advice_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = advice_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        #print("emb_shape: ",emb.shape)
        emb = self.dropout(emb)
        #print("emb_drop_shape: ",emb.shape)
        batch = emb.shape[0]
        emb = torch.reshape(emb,(batch,-1, self.emb_dim))
        #print("emb_reshape_shape: ",emb.shape)
        return emb

class Advice_Encoder(nn.Module):
    def __init__(self,vocab_size, word_embed_size, hidden_size, num_layers, drop_prob=0.2):
        super(Advice_Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, word_embed_size)
        
        self.lstm = nn.LSTM(input_size = word_embed_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        #self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
      
    
    def init_word_embedding(self, weight_init):
        assert weight_init.shape == (self.vocab_size, self.word_embed_size)
        self.embedding_layer.weight.data[:self.vocab_size] = weight_init
        
    def forward(self, advice_words):
        batch= advice_words.shape[0]
        hidden = self.init_hidden(batch)
        context = self.init_hidden(batch)
        advice_embed = self.embedding_layer(advice_words)        
        out, (h,_) = self.lstm(advice_embed, (hidden,context))
        #print("out GRU_shape: ",out.shape)
        h = torch.reshape(h,(batch,-1))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_()
        return hidden


class Sentence_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, drop_prob=0.2):
        super(Sentence_Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        #self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch= x.shape[0]
        hidden = self.init_hidden(batch)
        #print("hidden initialized_shape: ",hidden.shape)
        out, h = self.gru(x,hidden)
        #print("out GRU_shape: ",out.shape)
        h = torch.reshape(h,(batch,-1))
        #print("h_GRU_shape: ",h.shape)
        #out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden

class Action_Decoder(nn.Module):
    def __init__(self,img_emb_size, sent_emb_size, output_dim, dropout):
        super(Action_Decoder, self).__init__()
        #comb_embed = img_emb_size+sent_emb_size
        self.output = output_dim
        self.dropout =  nn.Dropout(dropout)
        self.out_emb1 = nn.Linear(img_emb_size+sent_emb_size, 512)
        self.out_emb2 = nn.Linear(512,128)
        self.out_emb3 = nn.Linear(128,output_dim)
        self.relu = nn.ReLU()
        
    def forward(self,x,y): 
        concat_vec = torch.cat((x,y),dim= 1)
        #print("concatenated_vector: ",concat_vec)
        decoded1 = self.out_emb1(concat_vec)
        #print("decoded1_vector: ",decoded1)
        drop_vec1= self.dropout(self.relu(decoded1))
        #drop_vec1= self.dropout(decoded1)
        decoded2 = self.out_emb2(drop_vec1)
        drop_vec2= self.dropout(self.relu(decoded2))
        #decoded2 = self.out_emb2(drop_vec2)
        out = self.out_emb3(drop_vec2)
        #print("out_shape: ", out.shape)
        
        return out