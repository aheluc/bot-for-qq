import jieba
import model
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
from random import randint
from random import shuffle
from random import choice
import re
import os
import pickle

criterion = nn.NLLLoss()


class Replier:
    
    def __init__(self, model_save_path, dictionary, data, model_name, num_layers, embedding_dim, hidden_dim):
        self.dict = dictionary
        self.data = data
        self.num_layers = num_layers
        self.train_data = None
        self.test_data = None
        self._model = model.ReplierModel(len(self.dict), embedding_dim, hidden_dim, self.num_layers)
        self.best_val_loss = None
        self.lr = 0.01
        self.prev_loss = None
        
        self.model_save_path = model_save_path
        self.model_name = model_name
        
        dict_file = os.path.join(self.model_save_path, self.model_name) + '.dict' 
        self.dict.save(dict_file)
        
    
    def _train(self):
        start_time = time.time()
        optimizer = optim.RMSprop(self._model.parameters(), lr=self.lr, weight_decay=0.0001)
        
        self._model.train(mode=True)
        
        batch_size = 165
        train_size = len(self.train_data)
        
        train_time = train_size / batch_size
        train_time = int(train_time) if train_time == int(train_time) else int(train_time) + 1
        print('%s batches.' % train_time)
        for batch_id in range(int(train_time)):
            self._model.zero_grad()
            loss = 0
            count = 0
            
            for sent_id in range(batch_id * batch_size, min((batch_id + 1) * batch_size, train_size)):
                hidden = self._model.init_hidden(self.num_layers)
                sentence = self.train_data[sent_id]
                data, targets = Variable(torch.LongTensor(sentence[:-1])), Variable(torch.LongTensor(sentence[1:]))
                output, hidden = self._model(data, hidden)
                loss += criterion(output, targets)
                count += 1
                
            loss = loss / count
            loss.backward()
            
            optimizer.step()
            print('batch id: %s' % batch_id)
    
    def _evaluate(self):
        self._model.eval()
        total_loss = 0
        
        for sentence in self.test_data:
            hidden = self._model.init_hidden(self.num_layers)
            data, targets = Variable(torch.LongTensor(sentence[:-1])), Variable(torch.LongTensor(sentence[1:]))
            output, hidden = self._model(data, hidden)
            total_loss += criterion(output, targets)
        return (total_loss / len(self.test_data))
    
    def train(self, epochs):
        print('training start')
        
        prev_loss = None
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            training_rate = 0.80
            shuffle(self.data)
            train_size = max(int(len(self.data) * training_rate), 1)
            
            self.train_data = self.data[:train_size]
            self.test_data = self.data[train_size:]
            
            self._train()
            val_loss = self._evaluate().data[0]
            print('loss: %s' % val_loss)
            
            if not self.best_val_loss or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save()
            
            # 学习率退火
            if self.prev_loss and val_loss > self.prev_loss:
                self.lr /= 2.6
            self.prev_loss = val_loss
            
            
            print('learning rate: ', self.lr)
            print('%s epoch finished.' % str(epoch + 1))
            elapsed = time.time() - epoch_start_time
            print('elapsed %s s.' % str(elapsed))
        
        self._model.train(mode=False)
        print('training finished')
        
    def load(self, fn):
        model_fn = '%s.pkl' % fn
        model_fn = os.path.join(self.model_save_path, model_fn)
        with open(model_fn, 'rb') as f:
            self._model = torch.load(f)

    def save(self):
        loss = str(self.best_val_loss)
        filename = '%s_(%s)' % (self.model_name, loss)
        filename = os.path.join(self.model_save_path, filename)
        with open('%s.pkl' % filename, 'wb') as f:
            torch.save(self._model, f)
