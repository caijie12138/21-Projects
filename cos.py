import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#训练数据个数
train_example = 10000
test_example = 1000
example_step = 20
sample_gap = 0.01

def generate_data(seq):
    '''
    根据前example_step个数据 预测第example_step+1个值
    '''
    X = []
    Y = []
    for i in range(len(seq)-example_step):
        X.append(seq[i:i+example_step])
        Y.append(seq[i+example_step])
    return np.array(X,dtype = np.float32),np.array(Y,dtype=np.float32)

test_start = train_example*sample_gap
test_end = test_start+test_example*sample_gap

train_x,train_y = generate_data(np.cos(np.linspace(0,test_start,train_example)))
test_x,test_y = generate_data(np.cos(np.linspace(test_start,test_end,test_example)))
lstm_size = 30
lstm_layers = 2
x = tf.placeholder(tf.float32,[None,example_step,1],name = 'train_x')
y = tf.placeholder(tf.float32,[None,1],name='train_y')
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)#包含lstm_size个单元
def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)

cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
out_puts,final_state = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
outputs = out_puts[:,-1]
#print(out_puts.shape)(?, 20, 30)
predictions = tf.contrib.layers.fully_connected(outputs,1,activation_fn=tf.tanh)
loss = tf.losses.mean_squared_error(y,predictions)
optimize = tf.train.AdamOptimizer(0.01).minimize(loss)
def get_batch(X,Y,batch_size=64):
    for i in range(0,len(X),batch_size):
        begin_i = i
        end_i = i+batch_size if (i+batch_size)<len(X) else len(X)
        yield X[begin_i:end_i],Y[begin_i:end_i]
epoch = 20
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./train_cos/", sess.graph)
    sess.run(tf.global_variables_initializer())
    it = 1
    for e in range(epoch):
        for batch_x,batch_y in get_batch(train_x,train_y):
            
            cost,_ = sess.run([loss,optimize],feed_dict={x:batch_x[:,:,None],y:batch_y[:,None],keep_prob:0.5})
            if it%100==0:
                print('Epochs:{}/{}'.format(e, epoch),
                      'Iteration:{}'.format(it),
                      'Train loss: {:.8f}'.format(cost))
            it+=1
    feed_dict = {x:test_x[:,:,None], keep_prob:1.0}
    results = sess.run(predictions, feed_dict=feed_dict)
    #plt.plot(results,'r', label='predicted')
    #plt.plot(test_y, 'g--', label='real sin')
    #plt.legend()
    #plt.show()
