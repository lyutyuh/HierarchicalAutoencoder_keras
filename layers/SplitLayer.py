# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.layers import *


class SplitLayer(Layer):
    def __init__(self, output_sentence_num=None, output_sentence_len=None,
                 delimiter=None, pad_word=None,cut_head=False, **kwargs):
        self.output_sentence_num = output_sentence_num
        self.output_sentence_len = output_sentence_len
        self._delimiter  = delimiter # index of <stop>
        self._pad_word = pad_word
        self.cut_head = cut_head
        super(SplitLayer, self).__init__(**kwargs)

    def call(self, x):        
                
        def single_article_split(par):
            bpad, rpad, nopad = par[0], par[1], par[2]
            
            bool_mask = K.tf.fill(K.tf.shape(nopad)[0:1], False)
            bool_mask = K.tf.logical_or(bool_mask, K.tf.equal(nopad, self._delimiter))
            
            bool_mask = K.tf.concat([bool_mask, [True]], axis=0)  
            bool_mask = K.tf.concat([[True], bool_mask], axis=0)      
            
            begin   = K.tf.squeeze(K.tf.where(bool_mask), axis = -1)
            sizes   = K.tf.subtract(begin[1:], begin[:-1])
            ful_len = K.tf.reduce_sum(sizes)
            sizes   = sizes[0:self.output_sentence_num-1]      # keep first #(self.output_sentence_num-1) sentences discovered
            cur_len = K.tf.reduce_sum(sizes)                 # calculate the length of the tail sentence = ful_len-cur_len
            tail    = [[0, ful_len-cur_len]]
            sizes   = K.tf.concat([sizes, [ful_len-cur_len]], axis=-1) # pad 1 number (the length of the tail sentence)
            zr_size = [[0,self.output_sentence_num- K.tf.shape(sizes)[0]]]
            sizes   = K.tf.pad(sizes, zr_size, 'CONSTANT', constant_values=0) # padding zeros
            
            chunks  = K.tf.split(rpad, sizes, num=self.output_sentence_num)
            
            for i in range(len(chunks)):
                chunks[i] = K.tf.cond(K.tf.equal(K.tf.shape(chunks[i])[0], 1), lambda: K.tf.fill(K.tf.shape(chunks[i]),self._pad_word), lambda: chunks[i])
                if self.cut_head:
                    chunks[i] = chunks[i][1:self.output_sentence_len]  # drop <start>
                else:
                    chunks[i] = chunks[i][:self.output_sentence_len]
                pad_split = [[0, self.output_sentence_len - K.tf.shape(chunks[i])[0]]]
                chunks[i] = K.tf.pad(chunks[i], pad_split, 'CONSTANT', constant_values=self._pad_word)                
            return K.tf.stack(chunks)
        
        padding = K.tf.fill((K.tf.shape(x)[0], 1), self._delimiter)
                
        right_padded = K.tf.concat([x, padding], axis = 1)
        padded = K.tf.concat([padding, x, padding], axis = 1)
        
        par = [padded, right_padded, x]
        
        otp = K.tf.map_fn(
            single_article_split, par,
            dtype=K.tf.int32
        ) 
        
        LEN_sentences = K.tf.count_nonzero(otp, [2])
        LEN_doc = K.tf.count_nonzero(LEN_sentences, [1])
        MASK = K.tf.sequence_mask(
                    lengths=LEN_sentences,
                    maxlen=self.output_sentence_len,
                    dtype=K.tf.float32,
                    name='mask_by_lengths'
                )
        
        return [otp, LEN_doc, MASK]
        
        
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_sentence_num, self.output_sentence_len), 
                (input_shape[0], 1), 
                (input_shape[0], self.output_sentence_num, self.output_sentence_len)]
    
    def get_config(self):
        config = {
            'output_sentence_num': self.output_sentence_num,
            'output_sentence_len': self.output_sentence_len,
            'delimiter': self._delimiter,
            'pad_word': self._pad_word,
            'cut_head': self.cut_head
        }
        base_config = super(SplitLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    