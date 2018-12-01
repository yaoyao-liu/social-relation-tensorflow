import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from train import TRAINER

FLAGS = flags.FLAGS

## Options
flags.DEFINE_bool('train', True, 'true to train, false to test')
flags.DEFINE_string('net_arch', 'vgg19', 'embedding network architecture')
flags.DEFINE_bool('double_stream_mode', True, 'true to use double stream framework, false to use single stream framework')
flags.DEFINE_bool('finetune_mode', True, 'true to finetune from pre-trained model, fasle to train from scratch')
flags.DEFINE_integer('epoch_num', 10, 'number of epochs')
flags.DEFINE_integer('batch_size', 10, 'number of samples trained in a single batch')
flags.DEFINE_integer('img_resize', 224, 'resize images to a specific resolution')
flags.DEFINE_integer('cls_num', 16, '5 for relation, 16 for domain')
flags.DEFINE_float('learning_rate', 0.00001, 'the learning rate')
flags.DEFINE_bool('shuffle_dataset', True, 'shuffle the dataset before training or not')
flags.DEFINE_bool('load_ckpt', False, 'load checkpoint or not')
flags.DEFINE_string('log_label', 'experiment_01', 'the label for ckpt saving')
flags.DEFINE_string('img_list1', './data_label_splits/example/single_body1_train_16.txt', 'the directory for the first image list')
flags.DEFINE_string('img_list2', './data_label_splits/example/single_body2_train_16.txt', 'the directory for the second image list')
flags.DEFINE_string('pretrain_model_dir', './pre_model.npy', 'the directory for the pre-trained model')


def main():
    trainer = TRAINER()
    trainer.build_model()
    if FLAGS.train:
        trainer.train()
    else:
        trainer.test()

if __name__ == "__main__":
    main()
