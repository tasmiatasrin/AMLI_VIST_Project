import nltk
import pickle
import argparse
from collections import Counter
# from pycocotools.coco import COCO
import re
from xlrd import open_workbook
import glob
import os
import fnmatch

actions = []

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def build_vocab_frogger(json, threshold):

    # Creates a vocab wrapper and add some special tokens.
    file = open('Sample_Training_Set/Rationalizations.txt', 'r')
    # .lower() returns a version with all upper case characters replaced with lower case characters.
    text = file.read().lower()
    file.close()
    # replaces anything that is not a lowercase letter, a space, or an apostrophe with a space:
    text = text.lower()
    # print(type(text))
    # exit(0)
    text = re.sub('[^a-z\ \']+', " ", text)
    words = list(text.split())
    # print words
    # exit(0)
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')



    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def build_vocab_frogger_turk(json, threshold):

    #open excel file and create a list of all the words in the rationalizations
    wb = open_workbook(args.caption_path)
    text = ""
    for sheet in wb.sheets():
        number_of_rows = sheet.nrows
        number_of_columns = sheet.ncols
        rationalizations = []
        items = []
        rows = []
        lengths = []
        max_length = 0
        counter = Counter()
        for row in range(1, number_of_rows):
            values = []
            line = sheet.cell(row,4).value
            actions.append(sheet.cell(row,2).value)
            tokens = nltk.tokenize.word_tokenize(line.lower())
            counter.update(tokens)
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')


    input_vocab = Vocabulary()

    input_vocab.add_word('<pad>')
    input_vocab.add_word('<start>')
    input_vocab.add_word('<end>')
    input_vocab.add_word('<unk>')

    # Adds the words to the vocabulary

    for i,word in enumerate(actions):
        input_vocab.add_word(word)
    for i, word in enumerate(words):
        vocab.add_word(word)
    print(vocab.word2idx['again'])
    return vocab,input_vocab

def build_vocab_input(input_vocab):
    words = []
    for dirpath, dirs, files in os.walk('./data/SymbRep(2)/Current'):
        for filename in fnmatch.filter(files, '*.txt'):
            with open(os.path.join(dirpath, filename)) as f:
                content = f.readlines()
                for k,line in enumerate(content):
                    nums = line.split()
                    for i,num in enumerate(nums):
                        words.append(str(num))
    for dirpath, dirs, files in os.walk('./data/SymbRep(2)/Next'):
        for filename in fnmatch.filter(files, '*.txt'):
            with open(os.path.join(dirpath, filename)) as f:
                content = f.readlines()
                for k,line in enumerate(content):
                    nums = line.split()
                    for i,num in enumerate(nums):
                        words.append(str(num))
    for i, word in enumerate(words):
        input_vocab.add_word(word)
    return input_vocab

def main(args):
    # vocab = build_vocab(json=args.caption_path,
    #                     threshold=args.threshold)
    vocab , input_vocab = build_vocab_frogger_turk(json=args.caption_path,
                        threshold=args.threshold)
    input_vocab = build_vocab_input(input_vocab)
    print(input_vocab)
    vocab_path = args.vocab_path
    print(input_vocab.word2idx)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(args.input_vocab_path, 'wb') as f:
        pickle.dump(input_vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='Turk_Master_File.xlsx', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab_frogger.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--input_vocab_path', type=str, default='./data/input_vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=1, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)