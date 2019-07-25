
from __future__ import print_function
import numpy as np
import tensorflow as tf
import string
from Func import dataprep
from Func.batch import BatchGenerator
from Func.embedding import batches2string
from Func.seq2seq import seq2seq
from Func.utils import accuracy, sample, random_distribution, sample_distribution, logprob
# from Func.testclass import test


url = 'http://mattmahoney.net/dc/'
filepath='text_data.zip'
train_ratio=0.8
batch_size=64
num_unrollings=14
num_nodes =256
decode_steps=14
num_steps = 20000
summary_frequency = 1000
#dropout=1.0

#data prepare
dataprep.maybe_download(url,filepath, 31344016)
train_text, valid_text, train_size, valid_size = dataprep.split_data(filepath,train_ratio)
vocabulary_size = len(string.ascii_lowercase) + 2 # size of "[a-z] + ' ' + #(end of sentence)" = 28
first_letter = ord(string.ascii_lowercase[0]) # unicode of 'a'
train_batches = BatchGenerator(train_text, batch_size, num_unrollings, first_letter, vocabulary_size)
valid_batches = BatchGenerator(valid_text, 1, num_unrollings, first_letter, vocabulary_size)
batches,output_batches=train_batches.next()
#print(batches2string(batches,first_letter))
#print(batches2string(output_batches,first_letter))


graph = tf.Graph()
with graph.as_default():

    '''
    Model definition and training
    '''
    ourMedel = seq2seq(vocabulary_size, num_nodes)

    # initialization of state and output
    saved_state=tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_output=tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    state=saved_state
    output=saved_output
    reset_state=tf.group(output.assign(tf.zeros([batch_size, num_nodes])),state.assign(tf.zeros([batch_size, num_nodes])))

    # initialization of weights and biases of Classifier 
    weight = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    bias = tf.Variable(tf.zeros([vocabulary_size]))

    # initialization of train input, decoder_inputs and output
    train_inputs=[]
    decoder_inputs=[]
    outputs=[]
    for i in range(num_unrollings):
        train_inputs.append(tf.placeholder(tf.float32,shape=[batch_size,vocabulary_size]))
        decoder_inputs.append(tf.placeholder(tf.float32,shape=[batch_size,vocabulary_size]))

    # Model definition
    output,state = ourMedel.encoder(train_inputs,output,state)
    outputs,output,state=ourMedel.training_decoder(decoder_inputs,output,state)
    with tf.control_dependencies([saved_state.assign(state),saved_output.assign(output),]):
        logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), weight, bias)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                                    labels=tf.concat(decoder_inputs, 0), logits=logits))
    
    # Define loss function and optimizer
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.AdamOptimizer()
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
    
    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    '''
    Sample inferencing
    '''
    sample_input=[]
    sample_outputs=[]

    for i in range(num_unrollings):
        sample_input.append(tf.placeholder(tf.float32,shape=[1,vocabulary_size]))
        
    sample_saved_state=tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
    sample_saved_output=tf.Variable(tf.zeros([1, num_nodes]), trainable=False)
    sample_output=sample_saved_output
    sample_state=sample_saved_state
    
    reset_sample_state = tf.group(
        sample_output.assign(tf.zeros([1, num_nodes])),
        sample_state.assign(tf.zeros([1, num_nodes])),

        )
    

    sample_output,sample_state=ourMedel.encoder(sample_input,sample_output,sample_state)
    sample_decoder_outputs,sample_output,sample_state=ourMedel.inference_decoder(sample_input[-1],num_unrollings,sample_output,sample_state, weight, bias)

    with tf.control_dependencies([sample_saved_output.assign(sample_output),sample_saved_state.assign(sample_state),]):
        for d in sample_decoder_outputs:
                sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(d, weight, bias))
                sample_outputs.append(sample_prediction)


with tf.Session(graph=graph) as session:
  global dropout
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches,output_batches = train_batches.next()
    feed_dict = dict()
    dropout=0.5
    
    for i in range(num_unrollings):
        #Feeding input from reverse according to https://arxiv.org/abs/1409.3215
        feed_dict[train_inputs[i]]=batches[i]
        feed_dict[decoder_inputs[i]]=output_batches[i]

        
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    #reset_state.run()

    if step % (summary_frequency ) == 0:
        dropout=1
        print('-'*80)
        print('Step '+str(step))
        print('Loss '+str(l))
        
        labels=np.concatenate(list(output_batches)[:])
#       print(characters(labels))
#       print(characters(predictions))
        print('Batch Accuracy: %.2f' % float(accuracy(labels,predictions)*100))
        num_validation = valid_size // num_unrollings
        reset_sample_state.run()
        sum_acc=0
        for _ in range(num_validation):
            valid,valid_output=valid_batches.next()
            valid_feed_dict=dict()
            for i in range(num_unrollings):
                valid_feed_dict[sample_input[i]]=valid[i]
            sample_pred=session.run(sample_outputs,feed_dict=valid_feed_dict)
            labels=np.concatenate(list(valid_output)[:],axis=0)
            pred=np.concatenate(list(sample_pred)[:],axis=0)
            sum_acc=sum_acc + accuracy(labels,pred)
        val_acc=sum_acc/num_validation
        print('Validation Accuracy: %0.2f'%(val_acc*100))
        print('Input Test String '+str(batches2string(valid,first_letter)))
        print('Output Prediction'+str(batches2string(sample_pred,first_letter)))
        print('Actual'+str(batches2string(valid_output,first_letter)))