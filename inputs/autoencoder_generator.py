# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import sys
import random
import numpy as np
from utils.rank_io import *
from layers import DynamicMaxPooling
import scipy.sparse as sp


class AutoencoderGenerator(object):
    def __init__(self, config):
        random.seed(12345)
        self.__name = 'AutoencoderGenerator'
        self.config = config
        self.data = config['data']
        
        self.keep_delimiter = config['keep_delimiter']
        self.docs = list(self.data.keys())
        self.batch_size = config['batch_size']
        self.data_maxlen = config['text_maxlen']
        self.fill_word = config['pad_word']
        self.is_train = config['phase'] == 'TRAIN'
        self.point = 0
        self.sentence_maxnum = config['sentence_maxnum']
        self.sentence_maxlen = config['sentence_maxlen']
        
        self._START_ = config['start_sent']
        self._END_   = config['end_sent']
        
        self.check_list = ['data', 'text_maxlen', 'keep_delimiter', 
                           'start_sent', 'end_sent',
                           'batch_size', 'vocab_size', 'pad_word', 
                           'phase','sentence_maxnum', 'sentence_maxlen']
        if not self.check():
            raise TypeError('[AutoencoderGenerator] parameter check wrong.')
        random.shuffle(self.docs)
        random.seed(12345)
        self.docs_train = set(random.sample(list(self.docs), int(len(self.docs)*0.8)))
        self.docs_test  = set(self.docs) - self.docs_train        
        self.docs_train, self.docs_test = list(self.docs_train), list(self.docs_test)
        print('size of train set: %d, size of test set: %d'%(len(self.docs_train), len(self.docs_test)))

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print('[%s] Error %s not in config' % (self.__name, e), end='\n')
                return False
        return True

    def get_batch(self, mode=None):
        assert mode in ['train', 'test'], 'key error: require mode to be train or test'
        curr_batch_size = self.batch_size
        X = np.zeros((curr_batch_size, self.data_maxlen), dtype=np.int32)
        X_decin = np.zeros((curr_batch_size, self.data_maxlen), dtype=np.int32)
        X_len = np.zeros((curr_batch_size,), dtype=np.int32)        
        X[:] = self.fill_word
        pool = None
        
        if mode == 'train':
            pool = self.docs_train
        else:
            pool = self.docs_test
            
        for i in range(curr_batch_size):
            d = random.choice(pool)
            doc_samp = self.data[d]
            if self.keep_delimiter:
                pass
            else:
                doc_samp = [_ind for _ind in doc_samp if _ind not in (self._START_, self._END_)]
            
            d_len = min(self.data_maxlen, len(doc_samp))
            X[i, :d_len], X_len[i], X_decin[i, :min(d_len+1, self.data_maxlen)]  = doc_samp[:d_len], d_len, [self._START_]+doc_samp[:min(d_len, self.data_maxlen-1)]
        return X, X_len, X_decin

    def get_batch_generator(self, mode=None):
        assert mode in ['train', 'test'], 'key error: require mode to be train or test'    
        while True:
            sample = self.get_batch(mode)
            if not sample:
                break
            X, X_len, X_decin = sample
            dummy_target = np.zeros(shape=(self.batch_size, self.sentence_maxlen*self.sentence_maxnum, 1), dtype='int32')
                
            if self.keep_delimiter:
                dummy_target = np.zeros(shape=(self.batch_size, self.data_maxlen, 1), dtype='int32')
                
            yield ({'single_doc':X, 
                    'single_doc_len':X_len, 
                    'decoder_inp':X_decin}, 
                    dummy_target)

    def reset(self):
        self.point = 0