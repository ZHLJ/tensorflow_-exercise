# encoding:utf-8

import tensorflow as tf
from numpy.random import RandomState

# test tf.reduce_sum
# x = tf.constant([[[1, 2],
#                   [3, 4],
#                   [5, 6],
#                   [7, 8]],
#                  [[9, 10],
#                   [11, 12],
#                   [13, 14],
#                   [15, 16]]])
# y = tf.reduce_sum(x, axis=2)
# with tf.Session() as sess:
#     y = sess.run(y)
#     print(y)

# 用tf实现一个简单的神经网络
batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 在shape中使用None，方便使用不同大小的batch，
# 例如在训练时，设置成比较小的batch，当数据集比较小时，在测试时，可以用全部的数据做测试
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播算法
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 随机生成数据来模拟数据集，x1+x2<1表示正样本(例如零件合格)，反之表示负样本
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("初始w1:")
    print(sess.run(w1))
    print("初始w2:")
    print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
        if i % 1000 == 0:
            # 每隔1000轮，计算在所有数据上的交叉墒，并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
    print("训练得到的w1:")
    print(sess.run(w1))
    print("训练得到的w2:")
    print(sess.run(w2))


