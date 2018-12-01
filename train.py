import numpy as np
import tensorflow as tf
import scipy.misc as scm
import os

from tqdm import trange
from utils import *
from tensorflow.python.platform import flags

from models.vgg16 import VGG16
from models.vgg19 import VGG19
from models.resnet50 import RESNET50
from models.resnet101 import RESNET101

FLAGS = flags.FLAGS

class TRAINER(object):
    def __init__(self):
        self.cls_num = FLAGS.cls_num
        self.epoch_num = FLAGS.epoch_num
        self.batch_size = FLAGS.batch_size
        self.img_resize = FLAGS.img_resize
        self.learning_rate = FLAGS.learning_rate
        self.shuffle_dataset = FLAGS.shuffle_dataset
        self.img_list1 = FLAGS.img_list1
        self.img_list2 = FLAGS.img_list2
        self.pretrain_model_dir = FLAGS.pretrain_model_dir
        self.load_ckpt = FLAGS.load_ckpt

        self.log_label = FLAGS.log_label + '_netarch_' + FLAGS.net_arch + '_batchsize_' + str(FLAGS.batch_size) + '_learningrate_' + str(FLAGS.learning_rate) + '_clsnum_' + str(FLAGS.cls_num)
        if FLAGS.shuffle_dataset:
            self.log_label += '_shuffled'
        if FLAGS.double_stream_mode:
            self.log_label += '_doublestream'
        else:
            self.log_label += '_singlestream'
        
        self.log_dir = os.path.join('log', self.log_label)
        self.tensorboard_dir = os.path.join(self.log_dir , 'tensorboard')
        self.ckpt_dir = os.path.join(self.log_dir , 'ckpt')

        if not os.path.exists('log'):
            os.mkdir('log')
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.tensorboard_dir):
            os.mkdir(self.tensorboard_dir)
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

        if FLAGS.net_arch=='vgg16':
            self.embedding_class = VGG16()
            self.embedding_net = self.embedding_class.forward_network
        elif FLAGS.net_arch=='vgg19':
            self.embedding_class = VGG19()
            self.embedding_net = self.embedding_class.forward_network
        elif FLAGS.net_arch=='resnet50':
            self.embedding_class = RESNET50()
            self.embedding_net = self.embedding_class.forward_network
        elif FLAGS.net_arch=='resnet101':
            self.embedding_class = RESNET101()
            self.embedding_net = self.embedding_class.forward_network

        self.sess = tf.Session()

    def checkpoint_load(self):
        print("[*] Loading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
            return True
        else:
            return False 
   
    def checkpoint_save(self, step, saver):
        print("[*] Saving checkpoint...")
        model_name = "relation_model"
        saver.save(self.sess,os.path.join(self.ckpt_dir, model_name),global_step=step)

    def summary(self):
        # summary writer
        self.writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)
        self.train_loss_sum = tf.summary.scalar('loss', self.loss)
        self.train_acc_sum = tf.summary.scalar('accuracy', self.acc)
        self.train_sum = tf.summary.merge([self.train_acc_sum, self.train_loss_sum])

    def softmaxloss(self, pred, label):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label))

    def softmaxacc(self, pred, label):
        return tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(label, 1))

    def relation_net(self, input1_, input2_, reuse=False):
        with tf.variable_scope("relation_net", reuse=reuse) as vs:
            net = tf.concat([input1_, input2_], 1)
            net = tf.contrib.layers.fully_connected(net, 4096, activation_fn=tf.nn.relu, scope='fc6')
            net = tf.contrib.layers.fully_connected(net, 4096, activation_fn=tf.nn.relu, scope='fc7')
            net = tf.contrib.layers.fully_connected(net, self.cls_num, activation_fn=tf.nn.relu, scope='fc8')
            output_ = tf.nn.softmax(net, name='softmax')

        variables = tf.contrib.framework.get_variables(vs)
        return output_, variables


    def build_model(self):

        self.label = tf.placeholder(tf.float32, None, name='label')
        self.img1 = tf.placeholder(tf.float32, shape=(None, self.img_resize, self.img_resize, 3), name='img1')
        self.img2 = tf.placeholder(tf.float32, shape=(None, self.img_resize, self.img_resize, 3), name='img2')
       
        if FLAGS.double_stream_mode:
            self.embedding1, self.vars1 = self.embedding_net(self.img1, scope="embedding_stream0", reuse=False)
            self.embedding2, self.vars2 = self.embedding_net(self.img2, scope="embedding_stream1", reuse=False)
            self.vars_embedding = self.vars1 + self.vars2
        else:
            self.embedding1, self.vars1 = self.embedding_net(self.img1, scope="embedding_stream0", reuse=False)
            self.embedding2, _ = self.embedding_net(self.img2, scope="embedding_net0", reuse=True)
            self.vars_embedding = self.vars1

        self.pred, self.vars_relation = self.relation_net(self.embedding1, self.embedding2, reuse=False)
        self.loss = self.softmaxloss(self.pred, self.label)
        self.acc = self.softmaxacc(self.pred, self.label)

        self.vars_all = self.vars_embedding + self.vars_relation
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.vars_all)
        self.sess.run(tf.global_variables_initializer())

    def train(self):

        PRINT_INTERVAL = 100

        print("[*] Start training...")

        if self.load_ckpt == True:
            self.checkpoint_load()

        img1_label_list, img1_dir_list = get_img_list(self.img_list1)
        img2_label_list, img2_dir_list = get_img_list(self.img_list2)

        batch_idxs = len(img1_dir_list) // self.batch_size
        num_list = range(len(img1_dir_list))
        count = 0
        self.summary()
        saver = tf.train.Saver(max_to_keep=10)
        if self.shuffle_dataset == True:
            np.random.shuffle(num_list)
            print("[*] Shuffle dataset...")


        for epoch_idx in range(self.epoch_num):
            print("Epoch: {}".format(epoch_idx+1))
            for idx in trange(batch_idxs):
                data_list1 = process_list(img1_dir_list, num_list[idx * self.batch_size : (idx+1) * self.batch_size])
                data_list2 = process_list(img2_dir_list, num_list[idx * self.batch_size : (idx+1) * self.batch_size])
                label_list = [img1_label_list[val] for val in data_list1]
                img1_255 = load_img(data_list1, self.img_resize)
                img2_255 = load_img(data_list2, self.img_resize)
                label = preprocess_label(label_list, self.cls_num)

                feed_dict={self.img1: img1_255, self.img2: img2_255, self.label: label}
                input_tensors = [self.acc, self.loss, self.train_sum, self.optimizer]

                train_acc, train_loss, summary, _ = self.sess.run(input_tensors, feed_dict)
                self.writer.add_summary(summary, count)
                count += 1

                if count % PRINT_INTERVAL == 0:
                    print('Step:' + str(count) + ' Loss:' + str(train_loss) + ' Acc:' + str(train_acc))

            self.checkpoint_save(count, saver)                        
        self.sess.close()


    def test(self):
        print("[*] Start testing...")
        self.checkpoint_load()

        test_acc_list = []
        test_loss_list = []

        img1_label_list, img1_dir_list = get_img_list(self.img_list1)
        img2_label_list, img2_dir_list = get_img_list(self.img_list2)

        batch_idxs = len(img1_dir_list) // self.batch_size

        for idx in trange(batch_idxs):
            data_list1 = img1_dir_list[idx * self.batch_size : (idx+1) * self.batch_size]
            data_list2 = img2_dir_list[idx * self.batch_size : (idx+1) * self.batch_size]
            label_list = [img1_label_list[val] for val in data_list1]
            img1_255 = load_img(data_list1, self.img_resize)
            img2_255 = load_img(data_list2, self.img_resize)
            label = preprocess_label(label_list, self.cls_num)
            feed_dict={self.img1: img1_255, self.img2: img2_255, self.label: label}
            input_tensors = [self.acc, self.loss]
            test_acc, test_loss = self.sess.run(input_tensors, feed_dict)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)

        test_acc_array = np.array(test_acc_list)
        mean_acc = np.mean(test_acc_array, 0)
        test_loss_array = np.array(test_loss_list)
        mean_loss = np.mean(test_loss_array, 0)

        print('Test Loss:' + str(mean_loss) + ' Test Acc:' + str(mean_acc)) 
                
        self.sess.close()



