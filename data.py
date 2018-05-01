import jieba
import os
import pickle
import re
from collections import Counter
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

HEADER = '^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{1,2}:[0-9]{2}:[0-9]{2}) .*\(([0-9]*)\)$'
HEADER_MAIL = '^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{1,2}:[0-9]{2}:[0-9]{2}) .*\<([^>]*)\>$'
HEADER = re.compile(HEADER)
HEADER_MAIL = re.compile(HEADER_MAIL)

jieba.load_userdict('D:\\Game\\ai_for_qq\\qq.dict')
print('userdict loaded.')

class Dictionary(object):
    def __init__(self, data):
        self.word2idx = {'<eos>': 0}
        self.idx2word = ['<eos>']
        self.counter = Counter()
        for sent in data:
            for word in sent:
                self.counter[word] += 1
        self.idx2word = list(self.counter.keys())
        self.update_word2idx()

    def update_word2idx(self):
        for idx, word in enumerate(self.idx2word):
            self.word2idx[word] = idx
    
    def filter_by_freq(self, freq, data):
        self.idx2word = [word for word in self.idx2word if self.counter[word] >= freq]
        self.update_word2idx()
        washed_data = []
        for sentence in data:
            remain_flag = True
            for word in sentence:
                if word not in self.idx2word:
                    remain_flag = False
                    break
            if remain_flag:
                washed_data.append(sentence)
        return washed_data
    
    def to_index(self, data):
        index_data = []
        for sentence in data:
            index_data.append([self.word2idx[word] for word in sentence])
        return index_data
    
    def __len__(self):
        return len(self.idx2word)
        
    def load(self, fn):
        self.idx2word = pickle.load(open(fn, 'rb'))
        self.word2idx = {word: index for index, word in enumerate(self.idx2word)}
    
    def save(self, fn):
        pickle.dump(self.idx2word, open(fn, 'wb'))

stopword = ['[图片]', '[表情]', '[QQ红包]', '[点头]', '@']
qq_filter = ['10000']
standalization = [
                     ('a仗', 'a杖'),
                     ('兔锅', '土锅'),
                     ('2333', '233'),
                     ('“', '"'),
                     ('”', '"'),
                     ('~', '～'),
                     ('?', '？'),
                     ('!', '！'),
                     ('(', '（'),
                     (')', '）'),
                     (',', '，'),
                     (re.compile('\.{1}'), '。'),
                     (re.compile('\.{2,}'), '...'),
                     (re.compile('23{2,}'), '233'),
                 ]


def standalize(data):
    for old, target in standalization:
        if type(old) == str:
            data = data.replace(old, target)
        else:
            data, _ = old.subn(target, data)
    return data


def get_train_data(filename_list, target_qq=[], min_len=4):
    for filename in filename_list:
        if not os.path.isfile(filename):
            return
    
    current_qq = '--'
    is_target = False
    chat_record = []
    for filename in filename_list:
        with open(filename, 'r', encoding='utf-8') as history:
            for line in history:
                line = line.strip().replace('\n', '')
                for word in stopword:
                    line = line.replace(word, '')
                line = line.lower()
                if line.find('http') >= 0:
                    continue
                header = HEADER.match(line)
                if not header:
                    header = HEADER_MAIL.match(line)
                if header:
                    _, current_qq = header.groups()
                    if not target_qq:
                        is_target = True if current_qq in qq_filter else False
                    else:
                        is_target = current_qq in target_qq
                elif is_target and line:
                    line = standalize(line)
                    record = list(jieba.cut(line)) + ['<eos>']
                    record = [word for word in record if word != ' ']
                    if len(record) <= min_len:
                        continue
                    chat_record.append(record)
                    if len(chat_record) >= 100000:
                        break
    
    dict = Dictionary(chat_record)
    
    return dict, chat_record


def dump_data(fn, data):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)


def write_data(data, filename, test=True):
    sep = ' | ' if test else ''
    with open(filename, 'w', encoding='utf8') as f:
        for sent in data:
            sent = sep.join(sent[:-1])
            f.write(sent + '\r')


def filter_by_freq(dict, data, freq):
    data_count = len(data)
    while True:
        data = dict.filter_by_freq(freq, data)
        dict = Dictionary(data)
        if data_count == len(data):
            return dict, data
        data_count = len(data)


