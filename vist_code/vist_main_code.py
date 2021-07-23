import argparse
from vist_methods import grab_features, preprocess_stories, vect_sentences
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from build_vocab_v2 import Vocabulary
from torchvision import transforms
from vist_data_loader import get_loader
#from frogger_dataset_preprocessed import FroggerDataset
import pickle
from PIL import Image
# from frogger_dataset_gen import FroggerDataset
import os
import torchvision.models as models
import numpy as np
import sys
import json
# import cv2
from advice_models import image_EncoderCNN, DecoderRNN, EncoderCNN
import random
from math import floor
import re
import json


# model parameters
# num_epochs = 101
#total_step = 50

#hidden_size = 1024
# hidden_size =512
# embed_size = 512
# image_size = 224
# batch_size =8



# with open("data/vocab_frogger_preprocessed.pkl", 'rb') as f:
#     in_vocab = pickle.load(f)
# with open("data/input_vocab.pkl", 'rb') as f:
#     image_vocab = pickle.load(f)
        
# current_image_dir = 'data/All_Images/Current_State/'






# we might need this

# def get_length(source_list):
#     length = len(source_list)
#     try:
#         length = list(source_list).index(0)
#     except:
#         length = len(source_list)
#     return length

# def get_sentences(word_ids):
#     batch = len(word_ids[0])
#     sent_len = len(word_ids)
#     sentences = []
#     for col in range(batch):
#         sentence = ""
#         for row in range(sent_len):
#             word = in_vocab.idx2word[int(word_ids[row][col])]
#             if word != '<end>' and word != '<pad>' and word != '<start>':
#                 sentence = sentence + " " + word
        
#         sentence = sentence.strip()
#         sentences.append(sentence)
#     return sentences




def weight_gloVe(target_vocab):
    # words = pickle.load(open(f'glove.6B/6B.300_words.pkl', 'rb'))
    # word2idx = pickle.load(open(f'glove.6B/6B.300_idx.pkl', 'rb'))
    # vectors = bcolz.open(f'glove.6B/6B.300.dat')[:]

    glove_dir = '/Users/javi/Downloads/example_NACME/vist_code/glove/'

    os.chdir(glove_dir)

    glove = pickle.load(open(f'6B_300_words_emb.pkl', 'rb'))

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0
    emb_dim = 300

    for i in range(matrix_len):
        all_words = list(target_vocab.keys())

        word = all_words[i]
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
    
    print("words_found: ", words_found)

    home_dir = '/Users/javi/Downloads/example_NACME/'

    os.chdir(home_dir)

    return weights_matrix

# how do we update this???
# where do the data_indices go
def update_data_loader(vist_dataset, data_indices):

    train_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
    
    val_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
    
    # shuffled_advice = [rationalization_token[i] for i in range(len(rationalization_token))]
    # #shuffled_advice = random.shuffle(new_list_rationalization)
    # random.shuffle(shuffled_advice)
    # print("len_shuffled_list: ", len(shuffled_advice))

    # split = int(floor((90.0/100)*len(shuffled_advice)))           
    # tr = slice(0,split)
    # tr_good_ids = good_ids[tr]
    # print("tr_good_ids: ",len(tr_good_ids))
        		
    # tr_indices = [0,split-1]
    # #print("tr_contents: ",tr)
    # te_indices = [split,len(shuffled_advice)-1]

    # te = slice(split,len(shuffled_advice))
    # te_good_ids = good_ids[te]
    # print("te_good_ids: ",len(te_good_ids))

    # training_rationalizations = shuffled_advice[tr]
    # print("train rationalizations_tokens: ",len(training_rationalizations))
    # testing_rationalizations = shuffled_advice[te]

    ## first need to create the data points with advice with its image id. then could be shuffled.

    #cur_training_images, cur_test_images = frogger_dataset_ob.load_images(current_image_dir, tr_good_ids, te_good_ids,tr_indices)

    train_data_loader = get_loader(train_sentences, train_image_features,
                                pickle_dir, home_dir, batch_size, vocab, train_transform,
                                num_workers=0)
    val_data_loader = get_loader(val_sentences, val_image_features,
                                pickle_dir, home_dir, batch_size, vocab, val_transform,
                                num_workers=0)

    return train_data_loader, val_data_loader



    

def main(args):

    # directories
    home_dir = '/Users/javi/Downloads/example_NACME/'
    train_pickle_dir = '/Users/javi/Desktop/AMLI/Capstone/1_story_dictionary_and_pickles/train_pickles/'
    val_pickle_dir = '/Users/javi/Desktop/AMLI/Capstone/1_story_dictionary_and_pickles/val_pickles/'
    dict_dir = '/Users/javi/Downloads/example_NACME/dictionaries/'

    # grab train, test, val data
    # grab vocabularies
    os.chdir(dict_dir)

    # open JSON file
    train_data = open('train_data.json') # train_story_ids
    val_data = open('val_data.json') # val_story_ids

    # vocabularies
    words2ids = open('words_to_ids.json') # words to ids
    ids2words = open('ids_to_words.json')  # ids to words

    os.chdir(home_dir)
    
    # returns JSON object as a dictionary
    train_story_data = json.load(train_data)
    val_story_data = json.load(val_data)
    in_vocab = json.load(words2ids) # do I need to seperate the vocab for test and val or can it be all in one vocabulary
    out_vocab = json.load(ids2words)

    # print(len(train_story_data),len(val_story_data),len(in_vocab), print(len(out_vocab)))

    # close file
    train_data.close
    val_data.close
    words2ids.close
    ids2words.close

    # grab the keys
    train_story_keys = [key for key in train_story_data.keys()]
    val_story_keys = [key for key in val_story_data.keys()]
    print(f'Number of story ids: Train:{len(train_story_keys)}, Val:{len(val_story_keys)}')

    # preprocess the train data
    #train_story_keys = preprocess_stories(home_dir, train_pickle_dir, train_story_data, train_story_keys)
    # grab the image features
    train_image_features, train_image_names = grab_features(home_dir, train_pickle_dir, train_story_data, train_story_keys)
    # vectorize sentences
    train_sentences = vect_sentences(train_story_data, train_story_keys, in_vocab)

    # preprocess the val_data
    #val_story_keys = preprocess_stories(home_dir, val_pickle_dir, val_story_data, val_story_keys)
    # grab the image features
    val_image_features, val_image_names = grab_features(home_dir, val_pickle_dir, val_story_data, val_story_keys)
    # vectorize sentences
    val_sentences = vect_sentences(val_story_data, val_story_keys, in_vocab)

    # change this
    # # val data
    # # train data
    # test_pickle_dir = '/Users/javi/Desktop/AMLI/Capstone/1_story_dictionary_and_pickles/pickles_test/'

    # # open JSON file
    # d = open('tsplit1_dictionary.json')
    # v = open('6_complete_vocabulary.json') # vocabulary

    # # returns JSON object as a dictionary
    # train_story_data = json.load(d)
    # vocab = json.load(v) # do I need to seperate the vocab for test and val or can it be all in one vocabulary

    # # close file
    # d.close
    # v.close

    # # grab the keys
    # train_story_keys = [key for key in train_story_data.keys()]
    # print(f'Number of story ids: {len(train_story_keys)}')

    # val data

    # batch_size = 8

    # vist_dataset = get_loader(train_image_features, train_sentences, train_transform, batch_size = 8,num_workers=0)

    # vist_dataset = get_loader(
    #     vocab = vocab, 
    #     sentences = train_sentences,
    #     images = train_image_features, 
    #     pickle_dir=train_pickle_dir, 
    #     home_dir = home_dir,
    #     batch_size=batch_size,
    #     num_workers=0)




    train_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])

    val_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
        
    # frogger_dataset_ob = FroggerDataset('Turk_Master_File_sorted.xlsx', in_vocab)
    # #tr_good_ids, te_good_ids, tr_indices, te_indices, training_rationalizations, testing_rationalizations = frogger_dataset_ob.load_data()
    # #all_good_ids, data_indices, all_good_rationalizations = frogger_dataset_ob.load_data()
    # all_good_ids, all_good_rationalizations, all_actions = frogger_dataset_ob.load_data()
    # cur_images = frogger_dataset_ob.load_images(current_image_dir, all_good_ids)

    # advices = []
    # images = []
    # actions = []

    #for i in range (0,len(all_good_rationalizations)):
        #dict_advice = {}
        #dict_advice['image_name'] = cur_images[i]
        #dict_advice['advice'] = all_good_rationalizations[i]

        #advice_dataset.append(dict_advice)

    # for i in range (0,len(all_good_rationalizations)):
    #     advices.append(all_good_rationalizations[i])
    #     images.append(cur_images[i])
    #     actions.append(all_actions[i])

    # output_act = open('actions_frogger.pkl', 'wb')
    # pickle.dump(actions, output_act)

    # output = open('advices_frogger.pkl', 'wb')
    # pickle.dump(advices, output)

    # output_img = open('images_frogger.pkl', 'wb')
    # pickle.dump(images, output_img)

    
        

    # Tasmia's code
    # train_data_loader, val_data_loader = update_data_loader (frogger_dataset_ob, all_good_ids, data_indices, all_good_rationalizations)


    # # print("good_ids: ",len(all_good_ids))
    # train_data_loader, val_data_loader = update_data_loader(
    #     train_sentences, train_image_features, pickle_dir, home_dir,
    #     batch_size, vocab, train_transform,num_workers=0)



    ##### This is where I stoppped, right before the model
    ### I think the code below is for the model


 
    #cur_training_images, cur_test_images = frogger_dataset_ob.load_images(current_image_dir, tr_good_ids, te_good_ids,tr_indices)
    

    train_data_loader = get_loader(
        in_vocab, train_image_features, train_sentences, train_image_names, train_transform,
        batch_size = args.batch_size, shuffle=True, num_workers=0)


    val_data_loader = get_loader(
        in_vocab, val_image_features, val_sentences, val_image_names, val_transform,
        batch_size = args.batch_size, shuffle=True, num_workers=0)

    # val_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(train_data_loader.dataset.vocab)
    pretrained_word_weight = weight_gloVe(in_vocab)
    pretrained_word_weight = torch.from_numpy(pretrained_word_weight).float()
    pretrained_word_weight = pretrained_word_weight.to(device)
    #encoder = image_EncoderCNN(args.img_feature_size).to(device)
    encoder= EncoderCNN(args.img_feature_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.img_feature_size, args.hidden_size, vocab_size).to(device)
    decoder.init_word_embedding(pretrained_word_weight)
        
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-5)

    total_train_step = len(train_data_loader)
    print("numbers of train images: ",total_train_step)
    total_val_step = len(val_data_loader)
    print("numbers of val images: ",total_val_step)



    #if os.path.exists('models/decoder-90.pth'):
        #decoder.load_state_dict(torch.load('models/decoder-90.pth'))
        #print('decoder Model loaded')
    #if os.path.exists('models/encoder-90.pth'):
        #encoder.load_state_dict(torch.load('models/encoder-90.pth'))       
        #print('encoder Model loaded')
        
    #num_epochs = 1
    train_losses = []
    val_losses = []
    for epoch in range(0, args.num_epochs):  
        results = []      
        inference_results = []
        total_train_loss = 0
                       
        # set decoder and encoder into train mode
        encoder.train()
        decoder.train()
        for i_step in range(0, total_train_step):
            # zero the gradients
            decoder.zero_grad()
            encoder.zero_grad()
    
            
            # Obtain the batch.
            img_name, images, captions = next(iter(train_data_loader))
            
            # make the captions for targets and teacher forcer
            captions_target = captions[:, 1:].to(device)
            captions_train = captions[:, :captions.shape[1]-1].to(device)

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            
            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions_train)
            
            # Calculate the batch loss
            loss = 0.0
            for sj, output_result in enumerate(zip(outputs, captions_target)):
                length = get_length(output_result[1])
                x = output_result[0]
                y = output_result[1]
                loss += criterion(x[:length,], y[:length])
            #loss = criterion(outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update the parameters in the optimizer
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss/total_train_step
        train_losses.append(avg_train_loss)
        # - - - Validate - - -
        # turn the evaluation mode on
        with torch.no_grad():
            # set the evaluation mode
            encoder.eval()
            decoder.eval()
            total_val_loss = 0
            for val_step in range(0, total_val_step):
                # get the validation images and captions
                img_name, val_images, val_captions = next(iter(val_data_loader))

                # define the captions
                captions_target = val_captions[:, 1:].to(device)
                captions_train = val_captions[:, :val_captions.shape[1]-1].to(device)

                # Move batch of images and captions to GPU if CUDA is available.
                val_images = val_images.to(device)

                # Pass the inputs through the CNN-RNN model.
                features = encoder(val_images)
                outputs = decoder(features, captions_train)
                output_ids = decoder.inference(features)
                cap_predicted = outputs.cpu().numpy()
                predicted_sentences = get_sentences(output_ids)
                
                for i in range (0, args.batch_size):
                    caption_str = ""
                    caption_grndtrth = ""
                    for j in range (0,len(cap_predicted[i])):
                        out_str = np.argmax(cap_predicted[i][j])
                        #print("out_str: ",out_str)
                        out_ = int(out_str)
                        #out_ = str(out_)
                        out_cap = in_vocab.idx2word[out_str]
                        caption_str = caption_str + " " + out_cap
                                            
                    for k in range (0,len(outputs[i])):
                        grnd_str = captions_target[i][k]
                        #print("grnd_str: ", grnd_str)
                        grnd_ = int(grnd_str)
                        #print("grnd_: ", grnd_)
                        #grnd_ = str(grnd_)
                        grnd_cap = in_vocab.idx2word[grnd_]
                        caption_grndtrth = caption_grndtrth + " " + grnd_cap
                    
                    image_name = img_name [i]
                    
                    
                    results.append({u'image name': image_name, u'generated caption': caption_str, u'ground_truth_cap': caption_grndtrth})
                    inference_results.append({u'image name': image_name, u'generated caption': predicted_sentences[i], u'ground_truth_cap': caption_grndtrth})

                    #print("generated caption: ", caption_str, " ground_truth_cap: ", caption_grndtrth)
                    #" ground_ans: ", true_ans)
                        

                # Calculate the batch loss.
                val_loss = criterion(outputs.view(-1, vocab_size), captions_target.contiguous().view(-1))
                total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss/total_val_step
            val_losses.append(avg_val_loss)
            
            # save the losses
            
            
            # Get training statistics.
        stats = 'Epoch [%d/%d], Training Loss: %.4f, Val Loss: %.4f' % (epoch, args.num_epochs, avg_train_loss, avg_val_loss)
            
        # Print training statistics (on same line).
        print('\r' + stats)
        #sys.stdout.flush()
                
        # Save the weights.
        if epoch % 20 == 0:
            print("\nSaving the model")
            torch.save(decoder.state_dict(), os.path.join('models/', 'decoder_gray_updt_loader-%d.pth' % epoch))
            torch.save(encoder.state_dict(), os.path.join('models/', 'encoder_gray_updt_loader-%d.pth' % epoch))
            my_advice = list(results)
            inference_advice = list(inference_results)
            json.dump(my_advice,open('results/advices_img_gray_updt_loader-epoch-%d.json' % epoch,'w'))  
            json.dump(inference_advice,open('results/inference_advice_img_gray_updt_loader-epoch-%d.json' % epoch,'w'))  
            train_data_loader, val_data_loader = update_data_loader (frogger_dataset_ob, all_good_ids, data_indices, all_good_rationalizations)
            
    
    np.save('results/train_losses', np.array(train_losses))
    np.save('results/val_losses', np.array(val_losses))       
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default='./models/' ,
    #                     help='path for saving trained models')

    # parser.add_argument('--vocab_path', type=str, default='./models/vocab.pkl',
    #                     help='path for vocabulary wrapper')
    # parser.add_argument('--train_image_dir', type=str, default='./data/train' ,
    #                     help='directory for resized train images')
    # parser.add_argument('--val_image_dir', type=str, default='./data/val' ,
    #                     help='directory for resized val images')
    # parser.add_argument('--train_sis_path', type=str,
    #                     default='./data/sis/train.story-in-sequence.json',
    #                     help='path for train sis json file')
    # parser.add_argument('--val_sis_path', type=str,
    #                     default='./data/sis/val.story-in-sequence.json',
    #                     help='path for val sis json file')
    # parser.add_argument('--log_step', type=int , default=20,
    #                     help='step size for prining log info')
    # parser.add_argument('--img_feature_size', type=int , default=1024 ,
    #                     help='dimension of image feature')
    
    parser.add_argument('--image_size', type=int, default=100 ,
                        help='size for input images')
    parser.add_argument('--img_feature_size', type=int , default=512,
                        help='dimension of image feature')
    parser.add_argument('--embed_size', type=int , default=300 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--max_words', type=int , default=40,
                        help='maximum number of words in a sentence')
    parser.add_argument('--num_epochs', type=int, default=301)
    parser.add_argument('--batch_size', type=int, default=8)

    
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    args = parser.parse_args()
    print(args)
    main(args)

# def test_adv_gen(img):
#     img_arr = cv2.imread(img)
#     #print("image_arr after cv2: ", img_arr.shape)
#     resized_img = cv2.resize(img_arr, (224,224))
#     x = np.reshape(resized_img, (3,224,224))
#     image_arr = torch.Tensor(x)
    
#     encoder = EncoderCNN(embed_size)
#     decoder = DecoderRNN( embed_size,hidden_size, vocab_size)
    
#     img_path = "models/encoder_drpout_v2-100.pth"
#     enocder.load_state_dict(torch.load(img_path))
#     advice_path = "models/decoder_drpout_v2-100.pth"
#     #model = DecoderRNN(image_embedding_size,dim_hidden,vocab_size)
#     decoder.load_state_dict(torch.load(advice_path))
    
#     encoded_img = encoder(image_arr)
#     decoded_adv = decoder(encoded_img)
#     decoded_adv = decoded_adv.cpu().numpy()
    
#     caption_str = ""
#     adv_list = []
#     for j in range (0,len(decoded_adv)):
#         out_str = np.argmax(cap_predicted[j])
#         #print("out_str: ",out_str)
#         out_ = int(out_str)
#         #out_ = str(out_)
#         out_cap = in_vocab.idx2word[out_str]
#         caption_str = caption_str + " " + out_cap
    
#     print("caption_str: ", caption_str)
    
#     return encoded_img, caption_str
 