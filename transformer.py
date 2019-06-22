# -*- coding: utf-8 -*-

"""

Created on Fri Aug 31 13:55:51 2018



@author: 61995

"""


import math
import time

begin = time.clock()

import csv

import pickle

import numpy as np

import operator

import tensorflow as tf

from example_preprocess import *

from modules import *

import os, codecs

from tqdm import tqdm


X,X_test,labs,labs_test,batch_index,batch_index_test,X_mask,X_test_mask = pre_data()
#X,X_test,labs,labs_test,batch_index,batch_index_test,X_mask,X_test_mask,_,_,_,_,_,_  = pre_data()



batch_size = 512

maxlen = 19

#lrs = 0.001

#lr_boundary = [20]

num_blocks = 1

dim_proj = 128

num_epochs = 11

num_heads = 1

dropout_rate = 0.25

sinusoid = False

n_items = np.unique(np.concatenate((X,labs.reshape((len(labs),1))),axis=1)).shape[0]

global_ = tf.Variable(tf.constant(0), trainable=False)

B = tf.get_variable('b', shape = [dim_proj,2*dim_proj], initializer=tf.contrib.layers.xavier_initializer())
H_1= tf.get_variable('H1', shape = [dim_proj,dim_proj], initializer=tf.contrib.layers.xavier_initializer())
H_2= tf.get_variable('H2', shape = [dim_proj,dim_proj], initializer=tf.contrib.layers.xavier_initializer())
V= tf.get_variable('V', shape = [dim_proj,1], initializer=tf.contrib.layers.xavier_initializer())
x = tf.placeholder(tf.int32, shape=(None, maxlen))

y = tf.placeholder(tf.int32, shape=(None, n_items))
mask = tf.placeholder(tf.float32, shape=(None, maxlen))
embed_table = tf.get_variable('wgembed', shape = [n_items, dim_proj], initializer=tf.contrib.layers.xavier_initializer())
#embed_table = tf.Variable(tf.random_uniform([n_items, dim_proj],maxval=math.sqrt(3)), name = 'wgembed')
lr = tf.placeholder(tf.float32, shape=[])

w_1 = tf.get_variable('w_1', shape = [], initializer=tf.contrib.layers.xavier_initializer())
#W_2 = tf.get_variable('W_2', shape = [], initializer=tf.contrib.layers.xavier_initializer())

with tf.variable_scope("encoder"):

    

    emb = tf.nn.embedding_lookup(embed_table, x)

    emb += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1]),

                                      vocab_size=maxlen, 

                                      num_units=dim_proj, 

                                      zero_pad=False, 

                                      scale=False,

                                      scope="emb_position_encode")



    emb = tf.layers.dropout(emb, rate=dropout_rate,noise_shape=[batch_size, maxlen, dim_proj], training=True)



    for i in range(num_blocks):

        with tf.variable_scope("num_blocks_{}".format(i)):

        ### Multihead Attention

            emb = multihead_attention(queries=emb, 

                                           keys=emb, 

                                           num_units=dim_proj, 

                                           num_heads=num_heads, 

                                           dropout_rate=dropout_rate,

                                           is_training=True,

                                           causality=False,
                                           
                                           scope="self_attention")

                        

                        ### Feed Forward

            emb = feedforward(emb, num_units=[4*dim_proj, dim_proj])
            
            emb_1 = emb[:,-1,:]
with tf.variable_scope("decoder"):
    alpha_shtore=[]
    for i in range(maxlen):
        tmp = tf.sigmoid((tf.add(tf.matmul(emb[:,i,:],H_1),tf.matmul(emb[:,-1,:],H_2))))#(128,100)
        res = tf.matmul(tmp,V)#(128,1)
        alpha_shtore.append(res)
    alpha_shtore = tf.transpose(tf.reshape(tf.stack(alpha_shtore),[maxlen,-1]))
    alpha_shtore=tf.nn.softmax(alpha_shtore*mask)*mask
    for i in range(maxlen):
        if i==0:
            output=(tf.multiply(emb[:,i,:],tf.reshape(alpha_shtore[:,i],[batch_size,1])))
        else:
            output_store=output
            output=tf.add(output_store,(tf.multiply(emb[:,i,:],tf.reshape(alpha_shtore[:,i],[batch_size,1]))))


            

emb = tf.transpose(emb,[0,2,1])



            #emb = tf.contrib.layers.fully_connected(emb,1)

#    emb = te.reduce_sum(emb,)


#emb_1 = tf.layers.dense(emb,1, kernel_initializer = tf.contrib.layers.xavier_initializer())
#emb_1 = tf.layers.dense(emb,1)




#emb_1= tf.reshape(emb_1,[batch_size,dim_proj])
#emb_1 = emb[:,:,-1]
#emb_2 = tf.layers.dense(emb,1, kernel_initializer = tf.contrib.layers.xavier_initializer())
#emb_2 = tf.sigmoid(tf.reduce_mean(emb,axis = -1))
#emb_c =tf.concat([emb_1,emb_2],1)

#combine=tf.matmul(B,emb_c,transpose_b = True)
#emb_2= tf.reshape(emb_2,[batch_size,dim_proj])
#emb_2 = emb_2[:,:,0]
W_1 = tf.sigmoid(w_1)
W_2 = 1 - W_1
combine = W_1*emb_1+W_2*output
#combine = emb_1

    #decode = tf.matmul(embed_table,S)

logit = tf.transpose(tf.matmul(embed_table,combine,transpose_b = True))

    #y_smoothed = label_smoothing(y)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=y))

train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)

            



    



show=[]
show1 = []



count1 = 1

sum_lost = 0

count_20 = 1


with tf.Session() as sess:    

    init = tf.global_variables_initializer()

    sess.run(init)
    saver = tf.train.Saver()
    saver.save(sess, './graph_TN',global_step=1)

    for step in range(num_epochs):

        if sum_lost/count_20 < 5.3 and step > 1:

            lr_input = 0.00005
            
            r = np.arange(len(X))
            np.random.shuffle(r)
            X_store = X[r,:]
            labs_store = labs[r]
            X = X_store
            labs = labs_store

        else:

            lr_input = 0.001

        for i in range(len(batch_index)):#len(batch_index)

            batch_xs = X[batch_index[i]]
            batch_mask = X_mask[batch_index[i]]
            batch_ys = labs[batch_index[i]].reshape([batch_size,1])

            mt = np.zeros([n_items,batch_size])

            mt[labs[batch_index[i]],range(batch_size)]=1

            mt=np.transpose(mt)

            sess.run(train_op, feed_dict={x: batch_xs,y: mt,lr: lr_input,mask: batch_mask, })

            if i % 20 == 0:

                lost = sess.run(cost, feed_dict={x: batch_xs,y: mt,lr: lr_input,mask: batch_mask, })

                sum_lost = sum_lost + lost

                

                #print(sess.run(embed_table))

                count_20+=1

                score = np.array(sess.run(logit, feed_dict={x: batch_xs,y: mt,lr: lr_input,mask: batch_mask, }))

                rank = score.argsort()[:,-20:]

                noc = 0
                nommr = 0

                for j in range(batch_size):

                    if batch_ys[j] in rank[j,:]:

                        noc += 1
                        nommr += 1 / (20 - np.where(rank[j,:]==batch_ys[j])[0])

                

                count1 += 1

                print([i,sum_lost/count_20,noc/batch_size,nommr/batch_size])

        sumnoc_test = 0
        summmr_test = 0
        

        for ii in range(len(batch_index_test)):#len(batch_index_test)

            batch_xs = X_test[batch_index_test[ii]]
            
            batch_mask =  X_test_mask[batch_index_test[ii]]
            
            mt = np.zeros([n_items,batch_size])

            batch_ys = labs_test[batch_index_test[ii]]

            mt[labs_test[batch_index_test[ii]],range(batch_size)]=1

            mt=np.transpose(mt)

            score = np.array(sess.run(logit, feed_dict={x: batch_xs,y: mt,mask: batch_mask, }))

            rank = score.argsort()[:,-20:]

            noc = 0
            nommr = 0

            for j in range(batch_size):

                if batch_ys[j] in rank[j,:]:

                    noc += 1
                    nommr += 1 / (20 - np.where(rank[j,:]==batch_ys[j])[0])
            sumnoc_test = sumnoc_test + noc
            summmr_test = summmr_test + nommr

        print('echo:')

        show.append(sumnoc_test/(batch_size*len(batch_index_test)))
        show1.append(summmr_test/(batch_size*len(batch_index_test)))
        saver.save(sess, './TN_2',global_step=5,write_meta_graph=False)
        A = np.array(sess.run(W_1, feed_dict={x: batch_xs,y: mt,}))
        B = np.array(sess.run(W_2, feed_dict={x: batch_xs,y: mt,}))

        print([step,show,show1])
        print([A,B])

         

            

   # show2 = sess.run(emb_1, feed_dict={x: batch_xs,y: batch_ys,})

end = time.clock()

total_running_time = end - begin

print('total running time is '+ str(total_running_time//3600)+'hours')



