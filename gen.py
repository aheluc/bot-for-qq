import model
import jieba
import os
import torch
from torch.autograd import Variable


class Replier:
    
    def __init__(self, model_save_path, dictionary, num_layers, embedding_dim, hidden_dim):
        self.dict = dictionary
        self.num_layers = num_layers
        self.ntokens = len(self.dict)
        # size of word embeddings
        # embedding_dim = 27
        # number of hidden units per layer
        # hidden_dim = 27
        self._model = model.ReplierModel(len(self.dict), embedding_dim, hidden_dim, self.num_layers)
        self.temperature = 0.6
        self._model.eval()
        self.model_save_path = model_save_path
        
    def load(self, fn):
        model_fn = '%s.pkl' % fn
        model_fn = os.path.join(self.model_save_path, model_fn)
        with open(model_fn, 'rb') as f:
            self._model = torch.load(f)
        
    def make_sentence(self, word, is_rand=True):
        usable = []
        stat = False
        word = list(jieba.cut(word))
        
        for w in word:
            if not stat and w in self.dict.idx2word:
                usable.append(w)
                stat = True
            elif stat and w not in self.dict.idx2word:
                break
            elif w in self.dict.idx2word:
                usable.append(w)
        
        if not usable:
            return ''
        
        if usable[0] == '你':
            usable[0] = '我'
        
        word = [self.dict.word2idx[word] for word in usable]
        if len(word) > 1:
            word = word[:-1]
        
        _input = Variable(torch.LongTensor(word), volatile=True)
        
        
        for i in range(20):
            hidden = self._model.init_hidden(self.num_layers)
            
            output, h = self._model(_input, hidden)
            if is_rand:
                output = torch.unsqueeze(output[-1], 0)
                word_weights = output.squeeze().data.div(self.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
            else:
                topv, topi = output.data.topk(1)
                word_idx = topi[0][0]
            
            if type(word_idx) == torch.LongTensor:
                word_idx = word_idx[0]
            if self.dict.idx2word[word_idx] == '<eos>':
                break
            word.append(word_idx)
            
            _input = Variable(torch.LongTensor(word), volatile=True)
        
        sentence = [self.dict.idx2word[idx] for idx in word]
        return sentence
