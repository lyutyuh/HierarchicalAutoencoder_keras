# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import sys
sys.path.append('..')
import numpy as np
from keras.engine import Layer
import keras.backend as K
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras import activations
from keras import initializers
from keras import optimizers
from keras.initializers import RandomUniform, TruncatedNormal
from keras.layers import Reshape, Embedding, Dot, Add, Multiply, \
    Input, Dense, Conv2D, CuDNNLSTM, Softmax, Lambda, \
    Bidirectional, Concatenate, add, multiply, Activation, Masking
from keras.optimizers import Adam
import recurrentshop
from recurrentshop import LSTMCell
import layers
from layers.Linear import Linear
from layers.SplitLayer import SplitLayer
from layers.Lengths2Mask import Lengths2Mask
from utils.utility import *
from utils.rank_io import *
from inputs.autoencoder_generator import AutoencoderGenerator
from losses.autoencoder_losses import loss_wrapper


class StandardAutoencoder(object):    
    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print(e, end='\n')
                print('Error %s not in config' % e, end='\n')
                return False
        return True
    
    def save_weights(self, SaveConfigPath, SaveWeightsPath):
        if SaveConfigPath is None or SaveWeightsPath is None:
            raise NameError('Empty parameters...')
        if self.model is None:
            raise AttributeError('self.model does not exist...')
        with open(SaveConfigPath, 'w') as fin:
            fin.write(self.model.to_json())
        self.model.save_weights(SaveWeightsPath)
        return
    
    def load_model(self, JsonConfigPath=None, ModelWeightsPath=None, override=False):
        if self.model is not None:
            assert override, 'self.model exists, check override...'    
        if JsonConfigPath is None or ModelWeightsPath is None:
            raise NameError('Empty parameters...')
        with open(JsonConfigPath) as fread:
            jsonconfig = fread.read(-1)
            loaded_mod = model_from_json(jsonconfig, 
                                         custom_objects={'Linear': layers.Linear.Linear,
                                                         'SplitLayer': layers.SplitLayer.SplitLayer, 
                                                         'LSTMCell': recurrentshop.cells.LSTMCell, 
                                                         'Lengths2Mask': layers.Lengths2Mask.Lengths2Mask})      
        loaded_mod.load_weights(ModelWeightsPath)
        self.model = loaded_mod
        self.EncoderModel = loaded_mod.get_layer('encoder')
        self.DecoderModel_onestep = loaded_mod.get_layer('word_decoder')        
        self._targets, self._mask = loaded_mod.get_layer('get_targets').output, loaded_mod.get_layer('get_mask_reduced_dim').output        
        return
    
    
    def compile_model(self):
        loss_function = loss_wrapper(self._targets, self._mask)
        self.model.compile(optimizer=self.optim, loss=loss_function)
        return
    
    def __init__(self, config):
        
        self.model  = None
        self.check_list = {'text_maxlen','sentence_maxnum','sentence_maxlen',
                           'hidden_size','delimiter', 'pad_word', 'unk_word', 
                            'start_sent', 'end_sent','vocab_size', 'embed_size',
                            'learning_rate'}
        self.config = config
        assert self.check(), 'parametre check failed'
        
        self.size   = self.config['hidden_size']
        
        self.Emb  = Embedding(self.config['vocab_size'], 
                             self.config['embed_size'],
                             trainable=True
                            )
        self.Splitlayer_keephead = SplitLayer(delimiter=self.config['delimiter'], 
                                     output_sentence_len=self.config['sentence_maxlen'],
                                     output_sentence_num=self.config['sentence_maxnum'],
                                     pad_word=self.config['pad_word'],
                                     cut_head=False,
                                     name='Split_Layer_keep_head'
                               )
        self.Splitlayer_cuthead = SplitLayer(delimiter=self.config['delimiter'], 
                                     output_sentence_len=self.config['sentence_maxlen'],
                                     output_sentence_num=self.config['sentence_maxnum'],
                                     pad_word=self.config['pad_word'],
                                     cut_head=True, 
                                     name='Split_Layer_cut_head'
                               )
        self.Sentence_reshape1D = Reshape((self.config['sentence_maxnum']*self.config['sentence_maxlen'], ),
                                          name='Sentence_reshape1D')
        
        self.Sentence_reshape2D = Reshape((self.config['sentence_maxnum'], self.config['sentence_maxlen'], self.config['embed_size'],), name='Sentence_reshape2D')
        self.Encoder_word = CuDNNLSTM(units=self.size, name='Encoder_word', return_state=True)
        self.Decoder_word_cell = LSTMCell(units=self.size, name='Decoder_word_cell')
        
        self.AttentionMapper = Linear(output_size=self.size, bias=True, bias_start=0.0, activation='tanh')
        self.Join = Dense(units=1, use_bias=False, name='Join')  # shape : [attention_vec_size]
        self.Exp = Lambda(lambda x: K.exp(x), name='Exp')
        self.Calcprob = Dense(units=self.config['vocab_size'], activation='softmax', name='Calcprob')
        self.ArgMax = Lambda(lambda x: K.argmax(x, axis=-1), dtype='int32')
        self.Printer = Lambda(lambda x: K.tf.Print(x, [x]))
        self.Identical = Lambda(lambda x: x, name='Identical')
        
        self.EncoderModel = None
        self.DecoderModel_onestep = None
        
        self._mask = None
        self._targets = None
        
        self.optim = optimizers.SGD(config['learning_rate'])
        return
    
    def build_encoder(self):        
        enc_inp = Input(name='single_doc', shape=(self.config['text_maxlen'],), dtype='int32')
        sentences_embeded = self.Emb(enc_inp)        
        
        doc_encode, doc_state_h, doc_state_c = self.Encoder_word(sentences_embeded)
        # doc_encode shape:(batchsize, self.size)
        
        return Model(inputs=enc_inp, 
                     outputs=[doc_encode, doc_state_h, doc_state_c],
                     name='encoder')
    

    def build_word_decoder(self):        
        dec_word_index_inp = Input(name='decoder_word_index_input', shape=(1, ))  # index of a decoded word
        word_decode = self.Emb(dec_word_index_inp)
        word_decode = Lambda(lambda x: K.squeeze(x, axis=1))(word_decode)
        dec_state_h_inp = Input(name='decoder_word_state_h_input', shape=(self.size, ))
        dec_state_c_inp = Input(name='decoder_word_state_c_input', shape=(self.size, ))
        
        cell_output, word_state_h, word_state_c = self.Decoder_word_cell([word_decode, dec_state_h_inp, dec_state_c_inp])
        
        probs = self.Calcprob(cell_output)
        pred_ind = self.ArgMax(probs)
        
        return Model(inputs=[dec_word_index_inp, 
                             dec_state_h_inp, 
                             dec_state_c_inp], 
                     outputs=[probs, 
                              pred_ind, 
                              word_state_h, 
                              word_state_c], 
                     name='word_decoder')

    def build(self):
        if self.model is not None:
            return
        enc_inp = Input(name='single_doc', shape=(self.config['text_maxlen'],), dtype='int32')
        
        doc_len = Input(name='single_doc_len', shape=(1, ), dtype='int32')
        self._mask = Lengths2Mask(maxlen=self.config['text_maxlen'], name='get_mask')(doc_len)
        self._mask = Lambda(lambda x: K.tf.squeeze(x, axis=1), name='get_mask_reduced_dim')(self._mask)
        
        self.EncoderModel = self.build_encoder()
        
        doc_encode, word_state_h, word_state_c = self.EncoderModel(enc_inp)
        # sent_state_h, sent_state_c are the state vectors to be fed into self.DecoderModel_onesent
        # inputs  single_doc
        # outputs [doc_encode, 
        #         sent_state_h, 
        #         sent_state_c
        #         ]
        
        #####################################################################################
        ## decoding ##
        
        self.DecoderModel_oneword = self.build_word_decoder()
        # inputs   [dec_word_index_inp, 
        #           dec_state_h_inp, 
        #           dec_state_c_inp], 
        # outputs  [probs, 
        #           pred_ind, 
        #           word_state_h, 
        #           word_state_c]
                
        
        
        dec_inp = Input(name='decoder_inp', shape=(self.config['text_maxlen'],), dtype='int32')
        
        self._targets = Lambda(lambda x: x, name='get_targets')(enc_inp)
        
        self._targets, self._mask, word_state_h = self.Identical([self._targets, self._mask, word_state_h])
              
        
        print('self._targets.shape', self._targets.shape, self._targets.dtype)
        print('self._mask.shape', self._mask.shape, self._mask.dtype)
        
        _preds = []
        
        
        
        for t in range(self.config['text_maxlen']):
            word_decode_index = Lambda(lambda x: x[:, t:t+1], dtype='int32', name='extract_index%d'%(t))(dec_inp)
            # word_decode_index shape : (batch_size, 1)
            probs, pred_ind, word_state_h, word_state_c = self.DecoderModel_oneword([word_decode_index, 
                                                                                     word_state_h, 
                                                                                     word_state_c])
            _preds.append(probs)
            pass

            
        
        _preds = Lambda(lambda x: K.tf.stack(x, axis=1))(_preds)
        print(_preds.shape)
        # ls = Lambda(lambda x: K.sparse_categorical_crossentropy(x[0], x[1]))([_targets, _pred])
        model = Model(inputs=[enc_inp, doc_len, dec_inp], outputs=_preds)
        self.model = model
        return


    
def main():
    import time    
    datapath ='/mnt/E/WORK/DATA/200k_news_text/category_classification/corpus_preprocessed.txt'
    dataset, _ = read_data(datapath)
    
    print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')
    text_maxlen = 150
    sentence_maxnum = 10
    sentence_maxlen = 30
    genr_auto = AutoencoderGenerator({'data': dataset, 'text_maxlen': text_maxlen, 
                                      'sentence_maxnum': sentence_maxnum,
                                        'sentence_maxlen': sentence_maxlen, 
                                      'keep_delimiter':False, 
                                        'start_sent':2, 'end_sent':3,
                                        'batch_size': 16, 'vocab_size': 50000,  'pad_word': 0,
                                        'phase': 'TRAIN'})
    genfun_train = genr_auto.get_batch_generator(mode='train')
    genfun_test = genr_auto.get_batch_generator(mode='test')
    # should print: size of train set: 143164, size of test set: 35791
    
    md = StandardAutoencoder({'text_maxlen':text_maxlen, 
                           'sentence_maxnum':sentence_maxnum, 
                           'sentence_maxlen':sentence_maxlen,
                           'hidden_size': 256,
                           'delimiter':3, 'pad_word':0, 'unk_word':1, 
                           'start_sent':2, 'end_sent':3,
                           'vocab_size':50000, 'embed_size':300,
                           'learning_rate': 0.1})
    
    md.build()
    md.compile_model()
    md.model.summary()
    
    training_begins = time.time()
    for i_e in range(1, 10):
        timer_start = time.time()
        if timer_start - training_begins >= 21600:
            break
        history_train = md.model.fit_generator(
                        genfun_train,
                        steps_per_epoch=200,
                        epochs=1,
                        shuffle=False,
                        verbose=1
                    )
        history_eval = md.model.evaluate_generator(genfun_test, steps=100)
        md.save_weights(SaveConfigPath='../cachedmodels/StandardAutoencoder.config.json', 
                        SaveWeightsPath='../cachedmodels/StandardAutoencoder.%d.weights'%(i_e))
        timer_end = time.time()
        with open('../cachedmodels/StandardAutoencoder.log', 'a') as logger:
            timestamp = '[' + str(timer_start) + ' ' + str(timer_end) + ']'
            logline = str(history_train.history) + ' ' + str(history_eval) + '\n'
            logger.write(timestamp + logline)
        
    return

if __name__ == '__main__':
    main()
    pass