import tensorflow as tf
from tensorflow.contrib import rnn

embeddings_batch = tf.constant(
    [[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
sents_len_batch = tf.constant([2, 1])
unit_size = 4
input_dim = 3
for i in range(3):
    with tf.variable_scope('parallel_lstm_' + str(i)):
        lstm_fw_cell = rnn.BasicLSTMCell(int(unit_size/2), forget_bias=1.0)
        lstm_bw_cell = rnn.BasicLSTMCell(int(unit_size/2), forget_bias=1.0)
        lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=0.5, output_keep_prob=0.5)
        lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=0.5, output_keep_prob=0.5)
        """output_fw, _ = tf.nn.dynamic_rnn(lstm_fw_cell,
                                         embeddings_batch,
                                         sequence_length=sents_len_batch,
                                         dtype=tf.float32)"""
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                    lstm_bw_cell,
                                                                    embeddings_batch,
                                                                    sequence_length=sents_len_batch,
                                                                    dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
        if i == 0:
            sentlayer_outputs = output
        else:
            sentlayer_outputs = tf.concat([sentlayer_outputs, output], axis=-1)
        """for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='parallel_lstm_'):
            print(v.name, v)"""
        fw_target_weight = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                              scope='parallel_lstm_' + str(i)+'/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')[0]
        print(fw_target_weight)
        fw_w_c = tf.reshape(tf.slice(fw_target_weight, begin=[0, 3 * int(unit_size/2)],
                                     size=[int(unit_size/2) + input_dim, int(unit_size/2)]), [-1])
        if i == 0:
            fw_parallel_w = fw_w_c
        else:
            fw_parallel_w = tf.concat([fw_parallel_w, fw_w_c], axis=-1)
        bw_target_weight = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                              scope='parallel_lstm_' + str(i) + '/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')[0]
        bw_w_c = tf.reshape(tf.slice(bw_target_weight, begin=[0, 3 * int(unit_size / 2)],
                                     size=[int(unit_size / 2) + input_dim, int(unit_size / 2)]), [-1])
        if i == 0:
            bw_parallel_w = bw_w_c
        else:
            bw_parallel_w = tf.concat([bw_parallel_w, bw_w_c], axis=-1)

fw_parallel_w = tf.reshape(fw_parallel_w, [-1, (int(unit_size / 2) + input_dim) * int(unit_size / 2)])
bw_parallel_w = tf.reshape(bw_parallel_w, [-1, (int(unit_size / 2) + input_dim) * int(unit_size / 2)])
parallel_w = tf.concat([fw_parallel_w, bw_parallel_w], axis=-1)
w_loss = tf.norm(tf.matmul(parallel_w, tf.transpose(parallel_w)) - tf.eye(3), ord='fro', axis=[-2, 1])

# variables_names =[v.name for v in tf.trainable_variables()]

# <tf.Variable 'rnn/basic_lstm_cell/kernel:0' shape=(8, 20) dtype=float32_ref>
# LSTM 有4个W（5*5）， 4个U（5*3或者3*5），将每个W或U的5拼到一起即4*5=20，这样既能放到一个矩阵里，形状即（3+5）*20
# <tf.Variable 'rnn/basic_lstm_cell/bias:0' shape=(20,) dtype=float32_ref>
"""for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='parallel_lstm_1'):
    print(v.name, v)
    if v.name == 'parallel_lstm_1/rnn/basic_lstm_cell/kernel:0':
        target_weight = v"""

"""w_c = tf.slice(target_weight, begin=[0, 15], size=[8, 5])
w_c = tf.reshape(w_c, [-1])"""
# print(w_c)
# v = tf.get_collection(lstm_fw_cell.trainable_variables)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    result = sess.run(sentlayer_outputs)
    """values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print(k, v)"""
    print(sess.run(parallel_w))
    print(sess.run(w_loss))
    print(result.shape)
