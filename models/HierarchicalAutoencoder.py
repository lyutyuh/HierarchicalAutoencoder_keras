# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import sys
sys.path.append('..')
from keras.engine import Layer
import keras.backend as K
from keras.models import Sequential, Model
from keras.models import model_from_json, model_from_yaml
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
from models.model import BasicModel
import layers
from layers.Linear import Linear
from layers.SplitLayer import SplitLayer
from utils.utility import *
from utils.rank_io import read_embedding, convert_embed_2_numpy
import numpy as np
from losses.autoencoder_losses import loss_wrapper

import tensorflow

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.Session(config = config)

class HierarchicalAttentionAutoencoder(object):    
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
                                                         'LSTMCell': recurrentshop.cells.LSTMCell})        
        loaded_mod.load_weights(ModelWeightsPath)
        self.model = loaded_mod
        self.EncoderModel = loaded_mod.get_layer('hierarchical_encoder')
        self.DecoderModel_onesent = loaded_mod.get_layer('sentence_decoder')
        self.DecoderModel_onestep = loaded_mod.get_layer('word_decoder')        
        self._targets, self._mask = loaded_mod.get_layer('Identical').get_output_at(0)[0], loaded_mod.get_layer('Identical').get_output_at(0)[1]
        
        return
    
    
    def compile_model(self):
        loss_function = loss_wrapper(self._targets, self._mask)
        self.model.compile(optimizer=self.optim, loss=loss_function)
        return
    
    def __init__(self, config):
        
        self.model  = None
        self.check_list = {'text_maxlen','sentence_maxnum','sentence_maxlen',
                           'hidden_size','delimiter', 'pad_word', 'unk_word', 
                                       'start_sent', 'end_sent',
                                       'vocab_size', 'embed_size',
                                       "embed_path",'embed_trainable', 
                           'learning_rate'}
        self.config = config
        assert self.check(), 'parametre check failed'
        
        self.size   = self.config['hidden_size']
        
        embed_dict = read_embedding(filename=self.config['embed_path'])
        self._PAD_ = self.config['pad_word']
        self._UNK_ = self.config['unk_word']
        self._START_ = self.config['start_sent']
        self._END_ = self.config['end_sent']
        embed_dict[self._PAD_] = np.zeros((self.config['embed_size'], ), dtype=np.float32)
        embed_dict[self._UNK_] = np.zeros((self.config['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.2, 0.2, [self.config['vocab_size'], self.config['embed_size']])) 
        weights = convert_embed_2_numpy(embed_dict, embed = embed)
        
        
        self.Emb  = Embedding(self.config['vocab_size'], 
                             self.config['embed_size'],
                             weights=[weights], 
                             trainable=self.config['embed_trainable']
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
        self.Encoder_word = CuDNNLSTM(units=self.size, name='Encoder_word')
        self.Encoder_sent = CuDNNLSTM(units=self.size, name='Encoder_sent', return_state=True)
        self.Decoder_word_cell = LSTMCell(units=self.size, name='Decoder_word_cell')
        self.Decoder_sent_cell = LSTMCell(units=self.size, name='Decoder_sent_cell')
        
        self.AttentionMapper = Linear(output_size=self.size, bias=True, bias_start=0.0, activation='tanh')
        self.Join = Dense(units=1, use_bias=False, name='Join')  # shape : [attention_vec_size]
        self.Exp = Lambda(lambda x: K.exp(x), name='Exp')
        self.Calcprob = Dense(units=self.config['vocab_size'], activation='softmax', name='Calcprob')
        self.ArgMax = Lambda(lambda x: K.argmax(x, axis=-1), dtype='int32')
        self.Printer = Lambda(lambda x: K.tf.Print(x, [x]))
        self.Identical = Lambda(lambda x: x, name='Identical')
        
        self.EncoderModel = None
        self.DecoderModel_onesent = None
        self.DecoderModel_onestep = None
        
        self._mask = None
        self._targets = None
        
        self.optim = optimizers.SGD(config['learning_rate'])
        return
    
    def build_encoder(self):        
        enc_inp = Input(name='single_doc', shape=(self.config['text_maxlen'],), dtype='int32')
        sentences = self.Splitlayer_keephead(enc_inp)[0]  # shape: (batchsize, sentence_maxnum, sentence_maxlen)
        sentences_reshaped1D = self.Sentence_reshape1D(sentences)
        
        sentences_embed = self.Sentence_reshape2D(self.Emb(sentences_reshaped1D))       
        
        Expand_dim_for_concat = Lambda(lambda x: K.expand_dims(x, 1))
        Concate_sentence_dim = Concatenate(axis=1)
        def encoding_words(_text):
            encoded_blocks = []
            for ind_sentence in range(self.config['sentence_maxnum']):
                target_sent = Lambda(lambda x: x[:,ind_sentence,:,:])(_text)
                encoded_blocks.append(Expand_dim_for_concat(self.Encoder_word(target_sent)))
            return Concate_sentence_dim(encoded_blocks)
        
        
        sent_encodes = encoding_words(sentences_embed)  
        # sent_encodes shape:(batchsize, sentence_maxnum, self.size)
        doc_encode, sent_state_h, sent_state_c = self.Encoder_sent(sent_encodes)
        # doc_encode shape:(batchsize, self.size)
        
        return Model(inputs=enc_inp, 
                     outputs=[doc_encode, 
                              sent_state_h, 
                              sent_state_c, 
                              sent_encodes],
                     name='hierarchical_encoder')
    
    def build_sent_decoder(self):        
        dec_sent_encodings_vec_inp = Input(name='decoder_sentence_encodings_vector_input', 
                                           shape=(self.config['sentence_maxnum'], self.size))
        prev_output     = Input(name='prev_output', shape=(self.size, ))
        dec_state_h_inp = Input(name='decoder_sent_state_h_input', shape=(self.size, ))
        dec_state_c_inp = Input(name='decoder_sent_state_c_input', shape=(self.size, ))
        
        vs = []
        for s2 in range(self.config['sentence_maxnum']):
            extracted = Lambda(lambda x: x[:, s2, :])(dec_sent_encodings_vec_inp)
            v = self.Join(self.AttentionMapper([prev_output, extracted]))
            vs.append(v)

        # calculating attention signal                
        vs = Concatenate(axis=-1)(vs)
        print('attention signal shape', vs.shape)
        vs = Lambda(lambda x: K.expand_dims(K.softmax(x, axis=-1), axis=-1), name='attention_weights')(vs)
        # vs = self.Printer(vs)
        context_vec = Lambda(lambda x: K.sum(x, axis=1))(Multiply()([vs, dec_sent_encodings_vec_inp]))

        sent_decode = Concatenate(axis=1)([prev_output, context_vec])
        cell_output, sent_state_h, sent_state_c = self.Decoder_sent_cell([sent_decode, dec_state_h_inp, dec_state_c_inp])
        
        return Model(inputs =[dec_sent_encodings_vec_inp,prev_output, dec_state_h_inp, dec_state_c_inp], 
                     outputs=[cell_output, sent_state_h, sent_state_c], 
                     name='sentence_decoder')
    
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
        
        self.EncoderModel = self.build_encoder()
        
        doc_encode, sent_state_h, sent_state_c, sent_encodings = self.EncoderModel(enc_inp)
        # sent_state_h, sent_state_c are the state vectors to be fed into self.DecoderModel_onesent
        # inputs  single_doc
        # outputs [doc_encode, 
        #         sent_state_h, 
        #         sent_state_c, 
        #         sent_encodes]
        
        #####################################################################################
        ## sentence decoding ##
        
        self.DecoderModel_onesent = self.build_sent_decoder()
        
        # inputs   [dec_sent_encodings_vec_inp, 
        #           prev_output,
        #           dec_state_h_inp, 
        #           dec_state_c_inp], 
        # outputs  [cell_output, sent_state_h, sent_state_c]
        
        self.DecoderModel_oneword = self.build_word_decoder()
        # inputs   [dec_word_index_inp, 
        #           dec_state_h_inp, 
        #           dec_state_c_inp], 
        # outputs  [probs, 
        #           pred_ind, 
        #           word_state_h, 
        #           word_state_c]
        dec_inp = Input(name='target_doc', shape=(self.config['text_maxlen'],), dtype='int32')
        dec_indices_by_sentence, _, _mask = self.Splitlayer_cuthead(dec_inp)
        
        self._targets = self.Sentence_reshape1D(dec_indices_by_sentence)
        self._mask    = self.Sentence_reshape1D(_mask)
        
        self._targets, self._mask, doc_encode = self.Identical([self._targets, self._mask, doc_encode])
        
        
        
        print('self._targets.shape', self._targets.shape, self._targets.dtype)
        print('self._mask.shape', self._mask.shape, self._mask.dtype)
        
        _preds = []        
        prev_sent_decoder_output  = Lambda(lambda x: K.zeros(shape=K.shape(x)))(doc_encode)
        
        for s1 in range(self.config['sentence_maxnum']):
            prev_sent_decoder_output, word_state_h, word_state_c = self.DecoderModel_onesent([sent_encodings, 
                                                                                             prev_sent_decoder_output,
                                                                                             sent_state_h, 
                                                                                             sent_state_c])           
            for t in range(self.config['sentence_maxlen']):
                word_decode_index = Lambda(lambda x: x[:, s1, t:t+1], dtype='int32')(dec_indices_by_sentence)
                probs, pred_ind, word_state_h, word_state_c = self.DecoderModel_oneword([word_decode_index, 
                                                                                         word_state_h, 
                                                                                         word_state_c])
                _preds.append(probs)
                pass

            sent_state_h, sent_state_c = word_state_h, word_state_c
        
        _preds = Lambda(lambda x: K.tf.stack(x, axis=1))(_preds)
        print(_preds.shape)
        # ls = Lambda(lambda x: K.sparse_categorical_crossentropy(x[0], x[1]))([_targets, _pred])
        model = Model(inputs=[enc_inp, dec_inp], outputs=_preds)
        
        self.model = model
        return
        