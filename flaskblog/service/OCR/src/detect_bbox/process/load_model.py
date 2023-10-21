# coding=utf-8
import os
import shutil
import tensorflow as tf

# sys.path.append(os.getcwd())
from nets import model_train as model
gpu = '0'
# checkpoint_path = 'checkpoints_mlt/'
# checkpoint_path1 = 'checkpoints_mlt/ctpn_50000.ckpt'
test_data_path = '/home/ncongthanh/Desktop/new_approval/new/text-detection-ctpn/data/demo/'
output_path = '/home/ncongthanh/Desktop/new_approval/new/text-detection-ctpn/data/res/'
checkpoint_path1 = '/home/ncongthanh/Desktop/new_approval/new/text-detection-ctpn/checkpoints_mlt/ctpn_50000.ckpt'
c = 0


# def load_model():

if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

with tf.get_default_graph().as_default():
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    bbox_pred, cls_pred, cls_prob = model.model(input_image)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # ckpt_state = tf.train.get_checkpoint_state(checkpoint_path1)
        # model_path = os.path.join(checkpoint_path1, os.path.basename(ckpt_state.model_checkpoint_path))
        model_path = checkpoint_path1
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)

    # return sess, bbox_pred, cls_prob, input_image, input_im_info