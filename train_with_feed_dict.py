import sys
sys.path.append('/Users/fvaggi/Dropbox/INRIA/Projects/deep_learning_fission_yeast/Code/models/')
import sugartensor as tf
import numpy as np
from sklearn.utils import shuffle
from itertools import cycle
from training_function import sg_train_func_with_dict

from tensorflow.examples.tutorials.mnist import input_data
mnist_data_dir = './asset/data/mnist'
reshape=False
one_hot=False
data_set = input_data.read_data_sets(mnist_data_dir, reshape=reshape, one_hot=one_hot)

def make_batch_iterator(input_tensors, dict_keys, batch_size=32, shuffle_tensors=True):
    #if shuffle_tensors:
    #    input_tensors = shuffle(input_tensors)

    # Crop to multiple of batch_size (for ease right now).
    n = input_tensors[0].shape[0]
    n_batches = n // batch_size

    def batch_iterator():
        for i in cycle(range(n_batches)):
            feed_dict = {}
            for j, key_name in enumerate(dict_keys):
                feed_dict[key_name] = input_tensors[j][i*batch_size:(i+1)*batch_size]
            yield feed_dict

    return batch_iterator

batch_size = 32   # batch size
num_category = 10  # total categorical factor
num_cont = 2  # total continuous factor
num_dim = 50  # total latent dimension

x = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1), name='digit_images')
y = tf.ones(batch_size, dtype=tf.float32)

# discriminator labels ( half 1s, half 0s )
y_disc = tf.concat(0, [y, y * 0])
real_labels = tf.placeholder(tf.int32, shape=(batch_size,), name='digit_labels')

batch_iterator = make_batch_iterator([data_set.train.images, data_set.train.labels.astype('int32')],
                                     [x, real_labels])

z_cat = tf.multinomial(tf.ones((batch_size, num_category), dtype=tf.sg_floatx) / num_category, 1).sg_squeeze().sg_int()

# random seed = random categorical variable + random uniform
z = z_cat.sg_one_hot(depth=num_category).sg_concat(target=tf.random_uniform((batch_size, num_dim - num_category)))

# random continuous variable
z_cont = z[:, num_category:num_category+num_cont]


with tf.sg_context(name='generator', size=4, stride=2, act='relu', bn=True):
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=7*7*128)
           .sg_reshape(shape=(-1, 7, 7, 128))
           .sg_upconv(dim=64)
           .sg_upconv(dim=1, act='sigmoid', bn=False))

# add image summary
tf.sg_summary_image(gen)



label = tf.concat(0, [real_labels, z_cat])

xx = tf.concat(0, [x, gen])

with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu'):
    # shared part
    shared = (xx.sg_conv(dim=64)
              .sg_conv(dim=128)
              .sg_flatten()
              .sg_dense(dim=1024))

    # discriminator end
    disc = shared.sg_dense(dim=1, act='linear').sg_squeeze()

    # shared recognizer part
    recog_shared = shared.sg_dense(dim=128)

    # categorical auxiliary classifier end
    class_cat = recog_shared.sg_dense(dim=num_category, act='linear')
    # continuous auxiliary classifier end
    class_cont = recog_shared[batch_size:, :].sg_dense(dim=num_cont, act='sigmoid')

#
# loss and train ops
#
loss_disc = tf.reduce_mean(disc.sg_bce(target=y_disc))  # discriminator loss
loss_gen = tf.reduce_mean(disc.sg_reuse(input=gen).sg_bce(target=y))  # generator loss
loss_class = tf.reduce_mean(class_cat.sg_ce(target=label)) \
             + tf.reduce_mean(class_cont.sg_mse(target=z_cont))  # recognizer loss

train_disc = tf.sg_optim(loss_disc + loss_class, lr=0.0001, category='discriminator')  # discriminator train ops
train_gen = tf.sg_optim(loss_gen + loss_class, lr=0.001, category='generator')  # generator train ops


# def alternate training func
@sg_train_func_with_dict
def alt_train(sess, opt):
    _feed_dict = opt.feed_dict
    try:
        l_disc = sess.run([loss_disc, train_disc], feed_dict=_feed_dict)[0]  # training discriminator
        l_gen = sess.run([loss_gen, train_gen], feed_dict=_feed_dict)[0]  # training generator
    except:
        _feed_dict
    return np.mean(l_disc) + np.mean(l_gen)


# Make data iterator:

# do training
alt_train(log_interval=10, max_ep=30, ep_size=10000, early_stop=False,
          batch_iterator=batch_iterator())