from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import re
import os
import pickle as pkl
import curses
import codecs

import numpy as np
import tensorflow as tf
tf.reset_default_graph()
from parser.base_network import BaseNetwork
from parser.neural import nn, nonlin, embeddings, recurrent, classifiers
from parser.structs import conllu_dataset
from parser.structs import vocabs
from parser.neural.optimizers import AdamOptimizer, AMSGradOptimizer

from tensorflow.python import debug as tf_debug
from parser.flag import Flag
import pdb
import time
#***************************************************************
class UnlabelGraphParserNetwork(BaseNetwork):
    """"""

    #--------------------------------------------------------------
    def build_graph(self, input_network_outputs={}, reuse=True):
        """"""
        # tf.set_random_seed(1234)
        # get input tensor
        with tf.variable_scope('Embeddings'):
            if self.sum_pos:  # TODO this should be done with a `POSMultivocab`
              pos_vocabs = list(filter(lambda x: 'POS' in x.classname, self.input_vocabs))
              pos_tensors = [input_vocab.get_input_tensor(embed_keep_prob=1, reuse=reuse) for input_vocab in pos_vocabs]
              non_pos_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs if
                                 'POS' not in input_vocab.classname]
              # pos_tensors = [tf.Print(pos_tensor, [pos_tensor]) for pos_tensor in pos_tensors]
              # non_pos_tensors = [tf.Print(non_pos_tensor, [non_pos_tensor]) for non_pos_tensor in non_pos_tensors]
              if pos_tensors:
                  pos_tensors = tf.add_n(pos_tensors)
                  if not reuse:
                      pos_tensors = [pos_vocabs[0].drop_func(pos_tensors, pos_vocabs[0].embed_keep_prob)]
                  else:
                      pos_tensors = [pos_tensors]
              input_tensors = non_pos_tensors + pos_tensors
            else:
                input_tensors = [input_vocab.get_input_tensor(reuse=reuse) for input_vocab in self.input_vocabs]

            for input_network, output in input_network_outputs:
              with tf.variable_scope(input_network.classname):
                  input_tensors.append(input_network.get_input_tensor(output, reuse=reuse))
            layer = tf.concat(input_tensors, 2)

        # get encoder output
        n_nonzero = tf.to_float(tf.count_nonzero(layer, axis=-1, keepdims=True))
        batch_size, bucket_size, input_size = nn.get_sizes(layer)
        layer *= input_size / (n_nonzero + tf.constant(1e-12))


        token_weights = nn.greater(self.id_vocab.placeholder, 0)
        tokens_per_sequence = tf.reduce_sum(token_weights, axis=1)
        n_tokens = tf.reduce_sum(tokens_per_sequence)
        n_sequences = tf.count_nonzero(tokens_per_sequence)
        seq_lengths = tokens_per_sequence + 1

        # root_weights = token_weights + (1 - nn.greater(tf.range(bucket_size), 0))
        root_weights = token_weights + (1 - tf.to_int64(nn.greater(tf.range(bucket_size), 0)))
        token_weights3D = tf.expand_dims(token_weights, axis=-1) * tf.expand_dims(root_weights, axis=-2)
        zero_diag = nn.zero_diag(layer)  # n x m x m with 0 in diag, others are 1
        tokens = {'n_tokens': n_tokens,
                  'tokens_per_sequence': tokens_per_sequence,
                  'token_weights': token_weights,
                  'token_weights3D': token_weights3D,
                  'n_sequences': n_sequences,
                  'bucket_size': bucket_size,
                  'zero_diag': zero_diag}

        conv_keep_prob = 1. if reuse else self.conv_keep_prob
        recur_keep_prob = 1. if reuse else self.recur_keep_prob
        recur_include_prob = 1. if reuse else self.recur_include_prob

        encoder_layer = layer
        for i in six.moves.range(self.n_encoder_layers):
            conv_width = self.first_layer_conv_width if not i else self.conv_width
            with tf.variable_scope('RNN-{}'.format(i)):
                encoder_layer, _ = recurrent.directed_RNN(encoder_layer, self.recur_size, seq_lengths,
                                                          bidirectional=self.bidirectional,
                                                          recur_cell=self.recur_cell,
                                                          conv_width=conv_width,
                                                          recur_func=self.recur_func,
                                                          conv_keep_prob=conv_keep_prob,
                                                          recur_include_prob=recur_include_prob,
                                                          recur_keep_prob=recur_keep_prob,
                                                          cifg=self.cifg,
                                                          highway=self.highway,
                                                          highway_func=self.highway_func,
                                                          bilin=self.bilin)


        output_fields = {vocab.field: vocab for vocab in self.output_vocabs}
        outputs = {}

        with tf.variable_scope('Classifiers'):

            if 'semrel' in output_fields:
                vocab = output_fields['semrel']
                head_vocab = output_fields['semhead']
                if vocab.factorized:
                    with tf.variable_scope('Unlabeled'):
                        unlabeled_outputs = head_vocab.get_bilinear_discriminator(
                            encoder_layer,
                            token_weights=token_weights3D,
                            reuse=reuse)
                    with tf.variable_scope('Labeled'):
                        labeled_outputs = vocab.get_bilinear_classifier(
                            encoder_layer, unlabeled_outputs.copy(),
                            token_weights=token_weights3D,
                            reuse=reuse)
                else:
                    labeled_outputs = vocab.get_unfactored_bilinear_classifier(layer, head_vocab.placeholder,
                                                                               token_weights=token_weights3D,
                                                                              reuse=reuse)
                outputs['encoder_edge_prob'] = unlabeled_outputs
                outputs['semgraph'] = labeled_outputs
                self._evals.add('semgraph')

            elif 'semhead' in output_fields:
                head_vocab = output_fields['semhead']
                # bilinear_discriminator: initial weights=0, logits=0
                unlabeled_outputs = head_vocab.get_bilinear_discriminator(encoder_layer,
                                                                          token_weights=token_weights3D,
                                                                          reuse=reuse)
                outputs['encoder_edge_prob'] = unlabeled_outputs


        # get decoder output

        decoder_rnn_layer = layer
        if self.decoder_with_lstm:
            for i in six.moves.range(self.n_decoder_layers):
                conv_width = self.first_layer_conv_width if not i else self.conv_width
                with tf.variable_scope('decoder_RNN-{}'.format(i)):
                    decoder_rnn_layer, _ = recurrent.UniRNN(decoder_rnn_layer, self.decoder_recur_size, seq_lengths,
                                                            recur_cell=self.decoder_recur_cell,
                                                            conv_width=conv_width,
                                                            recur_func=self.recur_func,
                                                            conv_keep_prob=conv_keep_prob,
                                                            recur_include_prob=recur_include_prob,
                                                            recur_keep_prob=recur_keep_prob,
                                                            cifg=self.cifg,
                                                            highway=self.highway,
                                                            highway_func=self.highway_func)


        assert self.decoder_MLP_dep_layers == len(self.hidden_sizes), 'You should give the number of hidden_sizes equal to number of layers'


        print_tensor = {}
        # print_tensor['input'] = layer
        # print_tensor['rnn_output'] = decoder_rnn_layer
        # print_tensor['loss_encoder_cross'] = unlabeled_outputs['loss']
        decoder_MLP_dep_layer = decoder_rnn_layer
        decoder_MLP_head_layer = layer

        for i in six.moves.range(1):#self.decoder_MLP_dep_layers):
            with tf.variable_scope('decoder_MLP_dep-{}'.format(i)):
                decoder_MLP_dep_layer = classifiers.hidden(decoder_MLP_dep_layer,
                                                           self.hidden_sizes[i],
                                                           hidden_func=tf.math.tanh,
                                                           hidden_keep_prob= self.hidden_keep_prob)
                # print_tensor['w_dep'] = w_dep
                # print_tensor['b_dep'] = b_dep
                # print_tensor['mlp_dep_output'] = decoder_MLP_dep_layer



        for i in six.moves.range(self.decoder_MLP_dep_layers):
            with tf.variable_scope('decoder_MLP_head-{}'.format(i)):
                decoder_MLP_head_layer = classifiers.hidden(decoder_MLP_head_layer,
                                                             self.hidden_sizes[i],
                                                             hidden_func=tf.math.tanh,
                                                             hidden_keep_prob=self.hidden_keep_prob)
                # print_tensor['w_head'] = w_head
                # print_tensor['b_head'] = b_head
                # print_tensor['mlp_head_output'] = decoder_MLP_head_layer

        #with tf.variable_scope('move_next'):
        vocabulary_size = 1
        input_fields = {vocab.field: vocab for vocab in self.input_vocabs}

        if 'form' in input_fields:
            if len(input_fields['form']) != 1:
                vocabulary_size = input_fields['form'].token_vocab.vocab_size
                token_index = input_fields['form'][0].placeholder
                token_index_next = nn.shift_left_by_one(token_index, dim=1)
            else:
                vocabulary_size = input_fields['form'].vocab_size
                token_index = input_fields['form'].placeholder
                token_index_next = nn.shift_left_by_one(token_index, dim=1)
        # print('**********************************')
        # print(vocabulary_size)
        # print('*********************************')

        token_index_next = tf.expand_dims(token_index_next, 1)
        token_index_next_expand = tf.tile(token_index_next, [1, bucket_size, 1])
        # ones = tf.ones_like(token_index_next)
        # token_index_next_expand = tf.expand_dims(ones, axis=-1) * tf.expand_dims(token_index_next, axis=-2)  # n x m x m

        # print_tensor['index'] = token_index_next
        # print_tensor['inexp'] = token_index_next_expand

        with tf.variable_scope('none_edge_softmax'):
          
            # zeros = tf.zeros_like(decoder_MLP_head_layer)
            # sum_zero_dep = nn.expand_add(decoder_MLP_dep_layer, zeros)
            # decoder_MLP_dep_layer_tanh = tf.math.tanh(sum_zero_dep, name='none_edge')
            #
            # logits_noedge = classifiers.hidden(decoder_MLP_dep_layer_tanh, vocabulary_size, hidden_func=tf.identity)
            # # print_tensor['noedge_score'] = logits_noedge
            # prob_none_edge = tf.nn.softmax(logits=logits_noedge) # n x m x m x V
            # print_tensor['noedge_softmax'] = prob_none_edge
            # prob_none_edge_ = nn.select_idx(prob_none_edge, token_index_next_expand) # n x m x m
            # print_tensor['noedge_selectprob'] = prob_none_edge_
            # prob_none_edge_ = nn.shift_right_by_one(prob_none_edge_, dim=2) # n x m x m
            # # print_tensor['noedge_selectprob'] = prob_none_edge_
            # prob_noedge = tf.transpose(prob_none_edge_, perm=[0,2,1])
            # outputs['decoder_noedge_prob'] = prob_noedge


            # decoder_MLP_dep_layer = tf.math.tanh(decoder_MLP_dep_layer, name='none_edge')
            logits = classifiers.hidden(decoder_MLP_dep_layer, vocabulary_size, hidden_func=tf.identity)
            prob_none_edge = tf.nn.softmax(logits=logits)  # n x m x V

            prob_none_edge = nn.select_idx(prob_none_edge, token_index_next)  # n x m
            prob_none_edge = nn.shift_right_by_one(prob_none_edge, dim=1)  # n x m
            # print_tensor['prob_noedge'] = prob_none_edge #tensor

            prob_none_edge = tf.expand_dims(prob_none_edge,1)
            prob_noedge = tf.tile(prob_none_edge,[1,bucket_size,1])


            # ones = tf.ones_like(prob_none_edge)
            # prob_noedge = tf.expand_dims(ones, axis=-1) * tf.expand_dims(prob_none_edge, axis=-2)# n x m x m
            # prob_noedge = nn.expand_dim_noedge(prob_none_edge)
            # print_tensor['prob_noedge_'] = prob_noedge
            prob_noedge = tf.transpose(prob_noedge, perm=[0, 2, 1])  # n x child x parent
            outputs['decoder_noedge_prob'] = prob_noedge


        with tf.variable_scope('with_edge_softmax'):
            
            #-------------------- (head+dep)W ---------------------------------#
            if self.d_sum:
                sum_head_dep = nn.expand_add(decoder_MLP_dep_layer, decoder_MLP_head_layer)
                # print_tensor['sum_head_dep'] = sum_head_dep
                sum_head_dep_tanh = tf.math.tanh(sum_head_dep, name='with_edge')

                logits_edge,_,_ = classifiers.hidden_w(sum_head_dep_tanh, vocabulary_size,
                                                     hidden_func=tf.identity)
                # print_tensor['withedge_score'] = logits_edge
                prob_with_edge_softmax = tf.nn.softmax(logits=logits_edge)  # n x m x m x V
                prob_with_edge_ = nn.select_idx(prob_with_edge_softmax, token_index_next_expand)  # n x m x m
                prob_with_edge_ = nn.shift_right_by_one(prob_with_edge_, dim=2)  # n x m x m
                # print_tensor['withedge_selectprob'] = prob_with_edge_
                prob_withedge = tf.transpose(prob_with_edge_, perm=[0, 2, 1])
            #-------------------------------------------------------------------#


            # # ------------------ headW+depW ---------------------------------#
            # dep_expand, head_expand = nn.expand_dim(decoder_MLP_dep_layer, decoder_MLP_head_layer)
            #
            # # print_tensor['child'] = decoder_MLP_dep_layer
            # # print_tensor['parent'] = decoder_MLP_head_layer
            # # print_tensor['child_ex'] = dep_expand
            # # print_tensor['parent_ex'] = head_expand
            #
            # concat = tf.concat([dep_expand,head_expand], 1)
            # # print_tensor['concat'] = concat
            #
            # with tf.variable_scope('1'):
            #     logits,w,b = classifiers.hidden_w(concat, vocabulary_size,
            #                                  hidden_func=tf.identity)
            #
            #     print_tensor['w'] = w
            #     print_tensor['b'] = b
            #
            # split = tf.split(logits,2,1)
            # dep_logits, head_logits = split[0], split[1]
            # # print_tensor['depl'] = dep_logits
            # # print_tensor['headl'] = head_logits
            #
            # sum_logits = dep_logits + head_logits
            #
            # print_tensor['dep_logits'] = dep_logits
            # print_tensor['head_logits'] = head_logits
            # print_tensor['sum_logits'] = sum_logits
            #
            #
            # prob_with_edge = tf.nn.softmax(logits=sum_logits) # n x m x m x V
            # prob_with_edge_ = nn.select_idx(prob_with_edge, token_index_next_expand)  # n x m x m
            # prob_with_edge_ = nn.shift_right_by_one(prob_with_edge_, dim=2)  # n x m x m
            # # print_tensor['withedge_selectprob'] = prob_with_edge_
            # prob_withedge = tf.transpose(prob_with_edge_, perm=[0, 2, 1])
            #
            # # # dep_prob = tf.nn.softmax(logits=dep_logits)
            # # dep_prob = nn.select_idx(dep_logits, token_index_next_expand)
            # # dep_prob = nn.shift_right_by_one(dep_prob,dim=2)
            # # dep_prob= tf.transpose(dep_prob,perm=[0,2,1])
            # #
            # # # head_prob = tf.nn.softmax(logits=head_logits)
            # # head_prob = nn.select_idx(head_logits, token_index_next_expand)
            # # head_prob = nn.shift_right_by_one(head_prob,dim=2)
            # # head_prob= tf.transpose(head_prob,perm=[0,2,1])
            # #
            # # print_tensor['dep_prob'] = dep_prob
            # # print_tensor['head_prob'] = head_prob
            # #---------------------------------------------------------------------#

            #---------------------------func(W[head,dep])---------------------------------#
            elif self.d_concat:
                # n x m x m x d, n x m x m x d
                decoder_MLP_dep_layer, decoder_MLP_head_layer = nn.expand_dims(decoder_MLP_dep_layer, decoder_MLP_head_layer)
                # n x m x m x 2d
                concat = tf.concat([decoder_MLP_dep_layer, decoder_MLP_head_layer], -1)
                with tf.variable_scope('nolinear'):
                #   n x m x m x d'
                    concat = classifiers.hidden(concat, 800, hidden_func=tf.math.tanh)
                with tf.variable_scope('score'):
                    logits = classifiers.hidden(concat, vocabulary_size, hidden_func=tf.identity) # n x m x m x V
                prob_with_edge_softmax = tf.nn.softmax(logits=logits)  # n x m x m x V
                prob_withedge = nn.select_idx(prob_with_edge_softmax, token_index_next_expand)  # n x m x m
                prob_withedge = nn.shift_right_by_one(prob_withedge, dim=2)  # n x m x m
                prob_withedge = tf.transpose(prob_withedge, perm=[0, 2, 1]) # n x m x m
            # ----------------------------------------------------------------------------#


            #-------------------------bilinear-refine---------------------------------#
            else:
            # with tf.device('/gpu:1'):
                logits = nn.expand_multiply(decoder_MLP_dep_layer, decoder_MLP_head_layer)
                with tf.variable_scope('score'):
                    logits = classifiers.hidden(logits, vocabulary_size, hidden_func=tf.identity) # n x m x m x V
                prob_with_edge_softmax = tf.nn.softmax(logits=logits)  # n x m x m x V
                prob_withedge = nn.select_idx(prob_with_edge_softmax, token_index_next_expand)  # n x m x m
                prob_withedge = nn.shift_right_by_one(prob_withedge, dim=2)  # n x m x m
                prob_withedge = tf.transpose(prob_withedge, perm=[0, 2, 1])  # n x m x m
            #--------------------------------------------------------------------------------#


            #------------------- bilinear:head*W*dep-------------------------------#
            # with tf.device('/gpu:1'):
            #     logits = classifiers.diagonal_bilinear_classifier(decoder_MLP_head_layer, decoder_MLP_dep_layer,
            #                                              vocabulary_size, hidden_keep_prob=1., add_linear=False)
            # logits = tf.transpose(logits,perm=[0,1,3,2])
            # # print_tensor['softmax'] = logits
            # prob_with_edge_softmax = tf.nn.softmax(logits=logits)  # n x m x m x V
            # # print_tensor['softmax'] = prob_with_edge
            # prob_with_edge = nn.select_idx(prob_with_edge_softmax, token_index_next_expand)  # n x m x m
            # prob_with_edge = nn.shift_right_by_one(prob_with_edge, dim=2)  # n x m x m
            # prob_withedge = tf.transpose(prob_with_edge, perm=[0, 2, 1])

            #----------------------------------------------------------------------#

            outputs['decoder_edge_prob'] = prob_withedge
            # print_tensor['withedge_prob'] = prob_withedge
            # print_tensor['noedge_prob'] = prob_noedge
            # print_tensor['with-no'] = prob_withedge-prob_noedge



        #--------------------------------------------------------------------------------
        with tf.variable_scope('loss'):


#             encoder_edge_prob = tf.reduce_sum(unlabeled_outputs['probabilities'],axis=3)
            encoder_edge_prob = unlabeled_outputs['probabilities']
            
            # print_tensor['encoder_prob'] = encoder_edge_prob

            #----- process decoder_prob, divide number of decoder lstm (number of head)----
            n_lstm = tf.to_float(bucket_size)
            prob_withedge = tf.pow(prob_withedge, 1 / n_lstm)
            prob_noedge = tf.pow(prob_noedge, 1 / n_lstm)
            #----- process decoder_prob, divide number of decoder lstm (number of head)----

            #------------Unsupervised loss-----------------------------------------
            # p(x|x)
            # loss_un = 0
            # if not self.supervised:
            # loss_un = -1/reduction*tf.reduce_sum(tf.math.log(nn.omit_zeros((encoder_edge_prob*prob_withedge
            #                                   + (1-encoder_edge_prob)*prob_noedge*tf.exp(self.restrict_loss_para))
            #                                    * (1-flag) *token_weights3D)))
            loss_un = (encoder_edge_prob*prob_withedge+(1-encoder_edge_prob)*prob_noedge*tf.exp(self.restrict_loss_para))


            # ------------Unsupervised loss-----------------------------------------


            # # p(x,y|x)
            #     max_pred = tf.to_float(nn.greater(encoder_edge_prob+prob_withedge,
            #                       1-encoder_edge_prob+prob_noedge+self.restrict_loss_para, dtype=tf.int64)
            #                       * token_weights3D)
            #
            #     loss = -tf.reduce_sum(tf.math.log(nn.omit_zeros(encoder_edge_prob*prob_withedge*max_pred)))\
            #        -tf.reduce_sum(tf.math.log(nn.omit_zeros((1-encoder_edge_prob)*prob_noedge*(1-max_pred))))


            #-----------Supervised loss-------------------------------------------

            # else:
            unlabeled_targets = tf.to_float(unlabeled_outputs['unlabeled_targets'])


            # loss_encoder = -1/reduction * tf.reduce_sum(tf.math.log(nn.omit_zeros(encoder_edge_prob*unlabeled_targets*flag*token_weights3D)))\
            #                -1/reduction * tf.reduce_sum(tf.math.log(nn.omit_zeros((1-encoder_edge_prob)*(1-unlabeled_targets)*flag*token_weights3D)))
            #
            # loss_decoder = -1/reduction * tf.reduce_sum(tf.math.log(nn.omit_zeros(prob_withedge*unlabeled_targets*flag*token_weights3D))) \
            #                + self.entropy_reg * tf.reduce_sum(prob_with_edge_softmax * tf.math.log(prob_with_edge_softmax+1e-12)) \
            #                -1/reduction * tf.reduce_sum(tf.math.log(nn.omit_zeros(prob_noedge*tf.exp(self.restrict_loss_para)*(1-unlabeled_targets)*flag*token_weights3D)))
            # loss_su = loss_encoder + loss_decoder
            # loss_su = -1/reduction * tf.reduce_sum(tf.math.log(nn.omit_zeros(encoder_edge_prob*prob_withedge*unlabeled_targets*flag*token_weights3D)))\
            #           -1/reduction * tf.reduce_sum(tf.math.log(nn.omit_zeros((1-encoder_edge_prob)*prob_noedge*(1-unlabeled_targets)*flag*token_weights3D)))
            loss_su = [encoder_edge_prob*prob_withedge*unlabeled_targets,(1-encoder_edge_prob)*prob_noedge*(1-unlabeled_targets)]
            entropy_loss = self.entropy_reg * tf.reduce_sum(prob_with_edge_softmax * tf.math.log(prob_with_edge_softmax+1e-12))
                # print_tensor['loss_encoder'] = loss_encoder
                # print_tensor['loss_decoder'] = loss_decoder
            # -----------Supervised loss-------------------------------------------

            outputs['loss_un'] = loss_un
            outputs['loss_su'] = loss_su
            outputs['entropy_loss'] = entropy_loss

        return outputs, tokens, print_tensor

    #--------------------------------------------------------------
    def train(self, load=False, noscreen=False):
        """"""

        trainset = conllu_dataset.CoNLLUTrainset(self.vocabs,
                                                 config=self._config)
        devset = conllu_dataset.CoNLLUDevset(self.vocabs,
                                             config=self._config)

        input_network_outputs = {}
        input_network_savers = []
        input_network_paths = []
        for input_network in self.input_networks:
            with tf.variable_scope(input_network.classname, reuse=False):
                input_network_outputs[input_network.classname] = input_network.build_graph(reuse=True)[0]
            network_variables = set(tf.global_variables(scope=input_network.classname))
            non_save_variables = set(tf.get_collection('non_save_variables'))
            network_save_variables = network_variables - non_save_variables
            saver = tf.train.Saver(list(network_save_variables))
            input_network_savers.append(saver)
            input_network_paths.append(self._config(self, input_network.classname + '_dir'))

        regularization_loss = self.l2_reg * tf.losses.get_regularization_loss() if self.l2_reg else 0
        with tf.variable_scope(self.classname, reuse=False):
            train_graph_outputs, tokens , print_tensor= self.build_graph(input_network_outputs=input_network_outputs, reuse=False)

            flag_obj = Flag(config=self._config)
            flag = flag_obj.placeholder
            flag = tf.expand_dims(flag, 1)
            flag = tf.tile(flag, multiples=[1, tokens['bucket_size']])
            flag = tf.expand_dims(flag, 1)
            flag = tf.tile(flag, multiples=[1, tokens['bucket_size'], 1])
            flag = tf.to_float(flag)

            token_weights3D = tf.to_float(tokens['token_weights3D'])
            reduction = tf.reduce_sum(token_weights3D)
            # # n x m x m
            # encoder_edge_prob = tf.math.sigmoid(train_graph_outputs['encoder_edge_prob']['logits'])
            # # n x m x m
            # decoder_edge_prob = train_graph_outputs['decoder_edge_prob']
            # # n x m x m
            # decoder_noedge_prob = train_graph_outputs['decoder_noedge_prob']
            #
            # # process decoder_prob, divide number of decoder lstm (number of head)
            # n_lstm = tf.to_float(tokens['bucket_size'])
            # decoder_edge_prob = tf.pow(decoder_edge_prob, 1/n_lstm)
            # decoder_noedge_prob = tf.pow(decoder_noedge_prob, 1/n_lstm)
            # # process decoder_prob, divide number of decoder lstm (number of head)
            #
            #
            # # number of without_edges in last prediction
            # number_noedge = tf.reduce_sum(nn.greater(decoder_noedge_prob + 1 - encoder_edge_prob,
            #                                          encoder_edge_prob + decoder_edge_prob))
            #
            # restrict_loss = tf.math.exp(number_noedge)
            #
            # #part_loss = tokens['zero_diag'] * tf.to_float(tokens['token_weights3D']) * \
            # part_loss = (encoder_edge_prob*decoder_edge_prob+(1-encoder_edge_prob)*decoder_noedge_prob)
            # train_loss = -tf.reduce_sum(tf.math.log(nn.omit_zeros(part_loss*tf.pow(restrict_loss,self.restrict_loss_para))))

            loss_un = tf.reduce_sum(tf.math.log(nn.omit_zeros(train_graph_outputs['loss_un']*(1-flag)*token_weights3D)))
            loss_su = tf.reduce_sum(tf.math.log(nn.omit_zeros(train_graph_outputs['loss_su'][0]*flag*token_weights3D)))\
                      +tf.reduce_sum(tf.math.log(nn.omit_zeros(train_graph_outputs['loss_su'][1]*flag*token_weights3D)))
            # ------------------ label loss -----------------------------------------------------------------
            loss_label_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_graph_outputs['semgraph']['label_targets'],
                                                                        logits=train_graph_outputs['semgraph']['label_logits'])
            loss_label = tf.reduce_sum(loss_label_tensor*flag*token_weights3D)

            train_loss = -1/reduction*(loss_su+ self.unsup_strength * loss_un-self.label_strength*loss_label)
            # -----------------label loss -------------------------------------------------------------------
            #train_loss = -1 / reduction * (loss_su + loss_un)
            loss = train_loss + regularization_loss + train_graph_outputs['entropy_loss']
            train_accuracy = self.compute_accuracy(train_graph_outputs, tokens, loss)

        with tf.variable_scope(self.classname, reuse=True):
            
            dev_graph_outputs, dev_tokens, print_tensor = self.build_graph(input_network_outputs=input_network_outputs, reuse=True)
            dev_flag_obj = Flag(config=self._config,isTrain=False,isDev=True)

            dev_loss = tf.constant(0)
            dev_accuracy = self.compute_accuracy(dev_graph_outputs, dev_tokens, dev_loss)
        dev_tensor = dev_accuracy

        #------------------------------------------------------------
        # test model
        # testset = conllu_dataset.CoNLLUTestset(self.vocabs,
        #                                      config=self._config)
        # with tf.variable_scope(self.classname, reuse=True):
        #
        #     test_graph_outputs, test_tokens, print_tensor = self.build_graph(input_network_outputs=input_network_outputs,
        #                                                                    reuse=True)
        #     test_flag_obj = Flag(config=self._config, isTrain=False)
        #
        #     test_loss = tf.constant(0)
        #     test_accuracy = self.compute_accuracy(test_graph_outputs, test_tokens, test_loss)
        # test_tensor = test_accuracy

        #------------------------------------------------------------

        update_step = tf.assign_add(self.global_step, 1)

        adam = AdamOptimizer(config=self._config)
        adam_op = adam.minimize(loss, variables=tf.trainable_variables(scope=self.classname)) # returns the current step
        adam_train_tensors = [print_tensor, adam_op, train_accuracy]
        # adam_train_tensors = [adam_op, train_accuracy]

        amsgrad = AMSGradOptimizer.from_optimizer(adam)
        amsgrad_op = amsgrad.minimize(loss, variables=tf.trainable_variables(scope=self.classname)) # returns the current step
        # amsgrad_train_tensors = [amsgrad_op, train_accuracy]
        amsgrad_train_tensors = [print_tensor, amsgrad_op, train_accuracy]

        if self.save_model_after_improvement or self.save_model_after_training:
            all_variables = set(tf.global_variables(scope=self.classname))
            non_save_variables = set(tf.get_collection('non_save_variables'))
            save_variables = all_variables - non_save_variables
            saver = tf.train.Saver(list(save_variables), max_to_keep=1)
            if self.save_model_fix_iter:
                saver = tf.train.Saver(list(save_variables), max_to_keep=0)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config) as sess:
            for saver, path in zip(input_network_savers, input_network_paths):
                saver.restore(sess, path)

            # debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)

            if self.load_model :
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, tf.train.latest_checkpoint('parser/UnlabelGraphParserNetwork'))

            else:
                sess.run(tf.global_variables_initializer())

            if noscreen: pass
            else:
                current_optimizer = 'Adam'
                train_tensors = adam_train_tensors
                current_step = 0
                print()
                print('\t', end='')
                print('{}\n'.format(self.save_dir), end='')
                print('\t', end='')
                print('GPU: {}\n'.format(self.cuda_visible_devices), end='')
                try:
                    current_epoch = 0
                    best_accuracy = 0
                    steps_since_best = 0

                    #--------------------------------------------------------


                    while (not self.max_steps or current_step < self.max_steps) and \
                            (not self.max_steps_without_improvement or steps_since_best < self.max_steps_without_improvement) and \
                            (not self.n_passes or current_epoch < len(trainset.conllu_files) * self.n_passes):
                        if steps_since_best >= 2500 and self.switch_optimizers and current_optimizer != 'AMSGrad':
                            train_tensors = amsgrad_train_tensors
                            current_optimizer = 'AMSGrad'
                            print('\t', end='')
                            print('Current optimizer: {}\n'.format(current_optimizer), end='')
                        train_history = self.init_history()
                        for batch in trainset.batch_iterator(shuffle=True):
                            # train_outputs.restart_timer()
                            # start_time = time.time()
                            # print (batch)
                            feed_dict = trainset.set_placeholders(batch)
                            feed_dict = flag_obj.set_placeholders(batch,feed_dict=feed_dict)
                            # pdb.set_trace()
                            # print('*********************************************************feed_dict')
                            # print(feed_dict)

                            print_tensor_, _, train_scores = sess.run(train_tensors, feed_dict=feed_dict)

                            # all_variables = tf.global_variables()
                            # print('************************************************************************')
                            # print(all_variables)
                            # print('************************************************************************')
                            # print(print_tensor_)
                            # # # #
                            # print('*****************with edge')
                            # print(train_scores['with_edge'])
                            # print("**********************no_edge")
                            # print(train_scores['no_edge'])
                            # print('*********************************************************predictions')
                            # print(train_scores['predictions'])
                            # print ()
                            # print('n_predictions: {}'.format(train_scores['n_predictions']))
                            # print ('n_targets:{}'.format(train_scores['n_targets']))
                            # print ('n_true_positives: {}'.format(train_scores['n_true_positives']))


                            self.update_history(train_history, train_scores)

                            current_step += 1

                            if current_step % self.print_every == 0:

                                dev_history = self.init_history()
                                for batch in devset.batch_iterator(shuffle=False):
                                    # dev_outputs.restart_timer()
                                    feed_dict = devset.set_placeholders(batch)
                                    feed_dict = dev_flag_obj.set_placeholders(batch,feed_dict=feed_dict)
                                    dev_scores = sess.run(dev_tensor, feed_dict=feed_dict)

                                    self.update_history(dev_history, dev_scores)

                                #current_accuracy *= .5
                                #current_accuracy += .5 * dev_outputs.get_current_accuracy()

                                current_loss_train = train_history['loss'][-1]

                                if dev_history['current_F1'] > best_accuracy:
                                    steps_since_best = 0
                                    best_accuracy = dev_history['current_F1']
                                    if self.save_model_after_improvement:
                                        saver.save(sess, os.path.join(self.save_dir, 'ckpt'), global_step=self.global_step,
                                                   write_meta_graph=False)

                                    # -------------------------------------------------
                                    # test model
                                    # test_history = self.init_history()
                                    # for batch in testset.batch_iterator(shuffle=False):
                                    #     # dev_outputs.restart_timer()
                                    #     feed_dict = testset.set_placeholders(batch)
                                    #     feed_dict = test_flag_obj.set_placeholders(batch, feed_dict=feed_dict)
                                    #     test_scores = sess.run(test_tensor, feed_dict=feed_dict)
                                    #
                                    #     self.update_history(test_history, test_scores)
                                    # ---------------------------------------------------


                                    if self.parse_devset:
                                        # self.parse_file(devset, dev_graph_outputs, dev_tokens, sess)

                                        self.parse_file(devset, dev_graph_outputs, dev_tokens, sess, output_dir='Parse-du-Train', output_filename='dev'+str(current_step))
                                        # self.parse_file(testset, test_graph_outputs, test_tokens, sess, output_dir='Parse-du-Train', output_filename='test'+str(current_step))

                                else:
                                    steps_since_best += self.print_every
                                current_epoch = sess.run(self.global_step)
                                print()
                                print('\t', end='')
                                print('Epoch: {:3d}'.format(int(current_epoch)), end='')
                                print(' | ', end='')
                                print('Step: {:5d}\n'.format(int(current_step)), end='')
                                print('\t', end='')
                                print('Current dev UF1: {:5.4f}'.format(dev_history['current_F1']), end='')
                                print(' | ', end='')
                                print('Current train UF1: {:5.4f}\n'.format(train_history['current_F1']), end='')
                                print('\t', end='')
                                print('Current dev Precision: {:5.4f}'.format(dev_history['current_precision']), end='')
                                print(' | ', end='')
                                print('Current train Precision: {:5.4f}\n'.format(train_history['current_precision']), end='')
                                print('\t', end='')
                                print('Current dev Recall: {:5.4f}'.format(dev_history['current_recall']), end='')
                                print(' | ', end='')
                                print('Current train Recall: {:5.4f}\n'.format(train_history['current_recall']), end='')
                                print('\t', end='')
                                #----------------------------------------------------------------------------
                                # test model
                                # print('Current test UF1: {:5.4f}'.format(test_history['current_F1']), end='')
                                # print(' | ', end='')
                                # print('Current test Recall: {:5.4f}'.format(test_history['current_recall']), end='')
                                # print(' | ', end='')
                                # print('Current test Precision: {:5.4f}\n'.format(test_history['current_precision']), end='')
                                #---------------------------------------------------------------------------
                                print('\t', end='')
                                print('Current best UF1: {:4f}\n'.format(best_accuracy), end='')
                                print('\t', end='')
                                print('Current train loss: {:4f}\n'.format(current_loss_train), end='')
                                print('\t', end='')
                                print('Steps since improvement: {:4d}\n'.format(int(steps_since_best)), end='')

                        # -----------------------------------------------------
                        # im = np.abs(last_loss - print_tensor_['loss_encoder'])
                        # last_loss = print_tensor_['loss_encoder']
                        # if im < 0.00000001:
                        #     break

                        current_epoch = sess.run(self.global_step)
                        sess.run(update_step)
                        trainset.load_next()
                    with open(os.path.join(self.save_dir, 'SUCCESS'), 'w') as f:
                        pass
                except KeyboardInterrupt:
                    pass

            # self.parse_file(devset, dev_graph_outputs, dev_tokens, sess)

            if self.save_model_after_training:
                saver.save(sess, os.path.join(self.save_dir, 'ckpt'), global_step=self.global_step, write_meta_graph=False)
        return

    #--------------------------------------------------------------
    def compute_accuracy(self, graph_outputs, tokens, loss):

        token_weights3D = tokens['token_weights3D']
        zero_diag = tf.to_int64(tokens['zero_diag'])

        encoder_edge_prob = tf.nn.sigmoid(graph_outputs['encoder_edge_prob']['logits'])
        # encoder_edge_score = graph_outputs['encoder_edge_prob']['logits']
        decoder_edge_prob = graph_outputs['decoder_edge_prob']
        decoder_noedge_prob = graph_outputs['decoder_noedge_prob']

        #-----process decoder_prob, divide number of decoder lstm (number of head)-----
        n_lstm = tf.to_float(tokens['bucket_size'])
        decoder_edge_prob = tf.divide(decoder_edge_prob, n_lstm)
        decoder_noedge_prob = tf.divide(decoder_noedge_prob, n_lstm)
        #-----process decoder_prob, divide number of decoder lstm (number of head)-----

        edge_score = encoder_edge_prob + decoder_edge_prob
        noedge_score = (1-encoder_edge_prob) + decoder_noedge_prob
        # (n x m x m) -> (n x m x m)
        predictions = nn.greater(edge_score, noedge_score, dtype=tf.int64) \
                      * token_weights3D * zero_diag
      
        # predictions = nn.greater(decoder_edge_prob, decoder_noedge_prob, dtype=tf.int64)*token_weights3D*tf.transpose(token_weights3D,perm=[0,2,1])

        # gold label
        unlabeled_targets = graph_outputs['encoder_edge_prob']['unlabeled_targets']
        # (n x m x m) (*) (n x m x m) -> (n x m x m)
        true_positives = predictions * unlabeled_targets
        # (n x m x m) -> () total prediction edge
        n_predictions = tf.reduce_sum(predictions)
        # (n x m x m) -> () total gold edge
        n_targets = tf.reduce_sum(unlabeled_targets)

        n_true_positives = tf.reduce_sum(true_positives)
        # () - () -> ()
        # n_false_positives = n_predictions - n_true_positives
        # n_false_negatives = n_targets - n_true_positives
        # n_edges = tf.to_float(tf.reduce_sum(token_weights3D))
        edge_precision = n_true_positives / n_predictions
        edge_recall = n_true_positives / n_targets
        edge_F1 = 2 * (edge_precision * edge_recall) / (edge_precision + edge_recall + 1e-12)
        accuracy = {}
        accuracy['loss'] = loss
        accuracy['predictions'] = predictions
        accuracy['n_predictions'] = n_predictions
        accuracy['n_targets'] = n_targets
        accuracy['n_true_positives'] = n_true_positives
        accuracy['edge_precision'] = edge_precision
        accuracy['edge_recall'] = edge_recall
        accuracy['edge_F1'] = edge_F1
        # accuracy['with_edge'] = decoder_edge_prob
        # accuracy['no_edge'] = decoder_noedge_prob

        return accuracy


    #--------------------------------------------------------------
    def init_history(self):
        history = {}
        history['loss'] = []
        history['loss'].append(100)
        history['precision'] = []
        history['recall'] = []
        history['F1'] = []
        history['current_n_predictions'] = 0
        history['current_n_targets'] = 0
        history['current_n_true_predictions'] = 0
        history['current_precision'] = 0
        history['current_recall'] = 0
        history['current_F1'] = 0
        return history

    #--------------------------------------------------------------
    def update_history(self, history, accuracy):
        history['loss'].append(accuracy['loss'])
        history['precision'].append(accuracy['edge_precision'])
        history['recall'].append(accuracy['edge_recall'])
        history['F1'].append(accuracy['edge_F1'])
        history['current_n_predictions'] += accuracy['n_predictions']
        history['current_n_targets'] += accuracy['n_targets']
        history['current_n_true_predictions'] += accuracy['n_true_positives']

        history['current_precision'] = history['current_n_true_predictions'] / (history['current_n_predictions']+ 1e-12)
        history['current_recall'] = history['current_n_true_predictions'] / history['current_n_targets']
        history['current_F1'] = 2 * (history['current_precision'] * history['current_recall']) / (history['current_precision'] + history['current_recall'] + 1e-12)

    #--------------------------------------------------------------
    '''
    def parse_file(self, dataset, graph_outputs, tokens, sess, output_dir=None, output_filename=None):

        input_filename = dataset.conllu_files[0]
        history = self.init_history()
        pred_dict = dict()
        for i, indices in enumerate(dataset.batch_iterator(shuffle=False)):
            tokens_, lengths = dataset.get_tokens(indices)
            feed_dict = dataset.set_placeholders(indices)
            # for field in graph_outputs:
            #     sess.run(tf.print(field))
            #     sess.run(tf.print(graph_outputs[field]), feed_dict=feed_dict)
            #     sess.run(tf.print('====='))
            loss = tf.constant(0.0)
            accuracy = sess.run(self.compute_accuracy(graph_outputs, tokens, loss), feed_dict=feed_dict)

            #             print(accuracy)
            self.update_history(history, accuracy)

            every_pred_dict = self.cache_predictions(tokens_, accuracy['predictions'], indices)
            for k in every_pred_dict.keys():
                if k in pred_dict.keys():
                    pred_dict[k].extend(every_pred_dict[k])
                else:
                    pred_dict[k] = every_pred_dict[k]

                    #         print(pred_dict)
        print(history['current_precision'], history['current_recall'], history['current_F1'])
        input_dir, input_filename = os.path.split(input_filename)
        if output_dir is None:
            output_dir = os.path.join(self.save_dir, 'parsed')
        if output_filename is None:
            output_filename = input_filename

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, output_filename)
        with codecs.open(output_filename, 'w', encoding='utf-8') as f:
            self.dump_current_predictions(pred_dict, f)

        return

        # ------------------------------------------------------------------

    def parse(self, conllu_files, output_dir=None, output_filename=None):

        parseset = conllu_dataset.CoNLLUDataset(conllu_files, self.vocabs,
                                                config=self._config)
        with tf.variable_scope(self.classname, reuse=False):
            parse_graph_outputs, tokens, _ = self.build_graph(reuse=True)
            #         parse_tensors = parse_outputs.accuracies
        all_variables = set(tf.global_variables())
        non_save_variables = set(tf.get_collection('non_save_variables'))
        save_variables = all_variables - non_save_variables
        saver = tf.train.Saver(list(save_variables), max_to_keep=1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            #             sess.run(tf.variables_initializer(list(non_save_variables)))
            saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))

            self.parse_file(parseset, parse_graph_outputs, tokens, sess, output_dir=output_dir,
                            output_filename=output_filename)
        return

        # ---------------------------------------------------------------

    def print_current_predictions(self, predictions):
        """
        print the predict sentences
        :return:
        """

        order = np.argsort(predictions['indices'])
        fields = ['form', 'lemma', 'upos', 'xpos', 'ufeats', 'dephead', 'deprel', 'semhead', 'misc']
        for i in order:
            j = 1
            token = []
            while j < len(predictions['id'][i]):
                token = [predictions['id'][i][j]]
                for field in fields:
                    if field in predictions:
                        token.append(predictions[field][i][j])
                    else:
                        token.append('_')
                print(u'\t'.join(token))
                j += 1
            print('')
        return

        # ------------------------------------------------------------------

    def cache_predictions(self, tokens, predictions, indices):
        """"""
        pred_dict = {}
        pred_dict['indices'] = []
        pred_dict['indices'].extend(indices)
        for field in tokens:
            if field not in pred_dict:
                pred_dict[field] = []
            pred_dict[field].extend(tokens[field])

        field_to_update = 'semhead'
        num_sentences = len(tokens[field_to_update])
        for s in range(num_sentences):
            num_words = len(tokens[field_to_update][s])
            for i in range(num_words):
                pred_dict[field_to_update][s][i] = ''
                for j in range(num_words):
                    if predictions[s][i][j]:
                        if np.any(predictions[s][i][j + 1:]):
                            pred_dict[field_to_update][s][i] += str(j) + ':ARG|'
                        else:
                            pred_dict[field_to_update][s][i] += str(j) + ':ARG'
                if not pred_dict[field_to_update][s][i]:
                    pred_dict[field_to_update][s][i] = '_'
        return pred_dict

        # ------------------------------------------------------------------

    def dump_current_predictions(self, predictions, f):
        """
        write to file
        :param f:
        :return:
        """
        #         print(predictions)
        order = np.argsort(predictions['indices'])
        fields = ['form', 'lemma', 'upos', 'xpos', 'ufeats', 'dephead', 'deprel', 'semhead', 'misc']
        for i in order:
            j = 1
            token = []
            while j < len(predictions['id'][i]):
                token = [predictions['id'][i][j]]
                for field in fields:
                    if field in predictions:
                        token.append(predictions[field][i][j])
                    else:
                        token.append('_')
                f.write('\t'.join(token) + '\n')
                j += 1
            f.write('\n')
        return
    '''

    def parse_file(self, dataset, graph_outputs, tokens, sess, output_dir=None, output_filename=None,time_ini=None):

        input_filename = dataset.conllu_files[0]
        # history = self.init_history()
        pred_dict = dict()
        pred_dict['indices'] = []
        for i, indices in enumerate(dataset.batch_iterator(shuffle=False)):
            tokens_, lengths = dataset.get_tokens(indices)
            feed_dict = dataset.set_placeholders(indices)
            # for field in graph_outputs:
            #     sess.run(tf.print(field))
            #     sess.run(tf.print(graph_outputs[field]), feed_dict=feed_dict)
            #     sess.run(tf.print('====='))
            loss = tf.constant(0.0)
            accuracy,labeled_pred = sess.run([self.compute_accuracy(graph_outputs, tokens, loss),graph_outputs['semgraph']], feed_dict=feed_dict)
            semrel_pred = np.argmax(labeled_pred['probabilities'], axis = -1)
            semrel_pred = semrel_pred * accuracy['predictions']
            #             print(accuracy)
            # self.update_history(history, accuracy)
            predictions = self.pred_to_sparse_pred(semrel_pred)
#             print(predictions)
            tokens_.update({vocab.field: vocab[predictions] for vocab in self.output_vocabs})
            #             print(tokens_)
            self.cache_predictions(tokens_, pred_dict, indices)

        input_dir, input_filename = os.path.split(input_filename)
        if output_dir is None:
            output_dir = os.path.join(self.save_dir, 'parsed')
        if output_filename is None:
            output_filename = input_filename

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, output_filename)
        with codecs.open(output_filename, 'w', encoding='utf-8') as f:
            self.dump_current_predictions(pred_dict, f)
        print('\033[92mParsing 1 file took {:0.1f} seconds\033[0m'.format(time.time() - time_ini))

        return

        # ------------------------------------------------------------------

    def parse(self, conllu_files, output_dir=None, output_filename=None):

        parseset = conllu_dataset.CoNLLUDataset(conllu_files, self.vocabs,
                                                config=self._config)
        with tf.variable_scope(self.classname, reuse=False):
            parse_graph_outputs, tokens, _ = self.build_graph(reuse=True)
            #         parse_tensors = parse_outputs.accuracies
        all_variables = set(tf.global_variables())
        non_save_variables = set(tf.get_collection('non_save_variables'))
        save_variables = all_variables - non_save_variables
        saver = tf.train.Saver(list(save_variables), max_to_keep=1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            #             sess.run(tf.variables_initializer(list(non_save_variables)))
            saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))
            time_ini = time.time()
            self.parse_file(parseset, parse_graph_outputs, tokens, sess, output_dir=output_dir,
                            output_filename=output_filename,time_ini = time_ini)
        return

        # ---------------------------------------------------------------

    def pred_to_sparse_pred(self, predictions):
        semgraph_preds = predictions
        sparse_semgraph_preds = []
        for i in range(len(semgraph_preds)):
            sparse_semgraph_preds.append([])
            for j in range(len(semgraph_preds[i])):
                sparse_semgraph_preds[-1].append([])
                for k, pred in enumerate(semgraph_preds[i, j]):
                    if pred:
                        sparse_semgraph_preds[-1][-1].append((k, semgraph_preds[i, j, k]))
        return sparse_semgraph_preds

    def print_current_predictions(self, predictions):
        """
        print the predict sentences
        :return:
        """

        order = np.argsort(predictions['indices'])
        fields = ['form', 'lemma', 'upos', 'xpos', 'ufeats', 'dephead', 'deprel', 'semrel', 'misc']
        for i in order:
            j = 1
            token = []
            while j < len(predictions['id'][i]):
                token = [predictions['id'][i][j]]
                for field in fields:
                    if field in predictions:
                        token.append(predictions[field][i][j])
                    else:
                        token.append('_')
                print(u'\t'.join(token))
                j += 1
            print('')
        return

        # ------------------------------------------------------------------

    def cache_predictions(self, tokens, predictions, indices):
        """"""
        predictions['indices'].extend(indices)
        for field in tokens:
            if field not in predictions:
                predictions[field] = []
            predictions[field].extend(tokens[field])
        return

        # ------------------------------------------------------------------

    def dump_current_predictions(self, predictions, f):
        """
        write to file
        :param f:
        :return:
        """

        order = np.argsort(predictions['indices'])
        fields = ['form', 'lemma', 'upos', 'xpos', 'ufeats', 'dephead', 'deprel', 'semrel', 'misc']
        for i in order:
            j = 1
            token = []
            while j < len(predictions['id'][i]):
                token = [predictions['id'][i][j]]
                for field in fields:
                    if field in predictions:
                        token.append(predictions[field][i][j])
                    else:
                        token.append('_')
#                 print(token)
                f.write('\t'.join(token) + '\n')
                j += 1
            f.write('\n')
        return


    @property
    def save_model_fix_iter(self):
        try:
            return self._config.getboolean(self, 'save_model_fix_iter')
        except:
            return False
    @property
    def decoder_with_lstm(self):
        try:
            return self._config.getboolean(self, 'decoder_with_lstm')
        except:
            return True

    @property
    def d_concat(self):
        try:
            return self._config.getboolean(self, 'd_concat')
        except:
            return False

    @property
    def d_sum(self):
        try:
            return self._config.getboolean(self, 'd_sum')
        except:
            return False
    @property
    def save_metadir(self):
        return self._config.getstr(self, 'save_metadir')

    @property
    def sum_pos(self):
        return self._config.getboolean(self, 'sum_pos')

    @property
    def load_model(self):
        return self._config.getboolean(self, 'load_model')

    @property
    def supervised(self):
        return self._config.getboolean(self, 'supervised')

    @property
    def n_encoder_layers(self):
        return self._config.getint(self, 'n_encoder_layers')
    @property
    def n_decoder_layers(self):
        return self._config.getint(self, 'n_decoder_layers')
    @property
    def decoder_recur_size(self):
        return self._config.getint(self, 'decoder_recur_size')
    @property
    def decoder_withedge_trans(self):
        return self._config.getint(self, 'decoder_withedge_trans')
    @property
    def decoder_MLP_dep_layers(self):
        return self._config.getint(self, 'decoder_MLP_dep_layers')
    @property
    def hidden_sizes(self):
        return self._config.getintlist(self, 'mlp_hidden_sizes')
    @property
    def hidden_keep_prob(self):
        return self._config.getfloat(self, 'hidden_keep_prob')

    @property
    def entropy_reg(self):
        return self._config.getfloat(self, 'entropy_reg')

    @property
    def decoder_recur_cell(self):
        recur_cell = self._config.getstr(self, 'decoder_recur_cell')
        if hasattr(recurrent, recur_cell):
            return getattr(recurrent, recur_cell)
        else:
            raise AttributeError("module '{}' has no attribute '{}'".format(recurrent.__name__, recur_cell))

    @property
    def restrict_loss_para(self):
        return self._config.getfloat(self, 'restrict_loss_para')
    
    @property
    def unsup_strength(self):
        return self._config.getfloat(self, 'unsup_strength')
    
    @property
    def label_strength(self):
        return self._config.getfloat(self, 'label_strength')
