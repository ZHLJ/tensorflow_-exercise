import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关的常数
INPUT_NODE = 784
OUTPUT_NODE = 10

# 配置神经网络参数
LAYER1_NODE = 500

BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


def inference_var(input_tensor, reuse=False):
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(weights))
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(biases))

    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        layer2 = tf.matmul(layer1, weights) + biases
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(weights))
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(biases))
    return layer2


def inference_var_avg(input_tensor, avg_class, reuse=True):
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases))
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(weights))
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(biases))
    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        layer2 = tf.matmul(layer1, avg_class.average(weights)) + avg_class.average(biases)
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(weights))
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(biases))
    return layer2


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 调用模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name='y-input')

    # # 生成隐藏层的参数
    # weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    # biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # # 生成输出层的参数
    # weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    # biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下的神经网络前向传播的结果
    # y = inference(x, None, weights1, biases1, weights2, biases2)
    y = inference_var(x, False)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    average_y = inference_var_avg(x, variable_averages, True)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    with tf.variable_scope('layer1', reuse=True):
        weights1 = tf.get_variable("weights")
    with tf.variable_scope('layer2', reuse=True):
        weights2 = tf.get_variable("weights")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization
    # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(cross_entropy_mean))
    # loss = tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g"
                      % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g"
              % (TRAINING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets("./data", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
