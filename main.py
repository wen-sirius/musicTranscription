# coding: utf8
"""
source code for the paper titled "Stereo feature enhancement and temporal information extraction network
for music transcription"

References:
[1] Xian Wang, Lingqiao Liu, and Qinfeng Shi, “Exploiting stereo sound channels to boost performance of neural network-based
music transcription,” in 18th IEEE International Conference On Machine Learning And Applications, ICMLA 2019, Boca Raton, FL, USA,
December 16-19, 2019, 2019, pp. 1353–1358
"""
from __future__ import print_function

import os

DEBUG = False  # in debug mode the numbers of recordings are minimized for fast debugging
GPU_ID = 3  # in case you have multiple GPUs, select the one to run this script

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops
tf.reset_default_graph()
import glob
import re
import librosa
import librosa.display
import numpy as np
from argparse import Namespace
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import datetime
from magenta.common import flatten_maybe_padded_sequences
import collections
import warnings
from tools import MiscFns
from MusicNet import MusicNet

# contain all configurations
class Config(object):
    def __init__(self):
        self.debug_mode = DEBUG
        self.test_with_30_secs = False
        self.gpu_id = GPU_ID

        self.num_epochs = 50
        self.batches_per_epoch = 5000
        self.batch_size = 4
        self.learning_rate = 1e-4

        self.train_or_inference = Namespace(
            inference=None,
            from_saved=None,
            model_prefix='net'
        )
        # inference: point to the saved model for inference
        # from_saved: point to the saved model from which the training continues
        # model_prefix: the prefix used when saving the model
        # order: If inference is not None, then do inference; elif from_saved is not None, then continue training
        #        from the saved model; elif train from scratch.
        #        If model_prefix is None, the model will not be saved.

        self.tb_dir = 'tb_inf'
        # the directory for saving tensorboard data including performance measures, model parameters, and the model itself

        # check if tb_dir exists
        #assert self.tb_dir is not None
        tmp_dirs = glob.glob('./*/')
        tmp_dirs = [s[2:-1] for s in tmp_dirs]
        if self.tb_dir in tmp_dirs:
            raise EnvironmentError('\n'
                                   'directory {} for storing tensorboard data already exists!\n'
                                   'Cannot proceed.\n'
                                   'Please specify a different directory.'.format(self.tb_dir)
                                   )

        # check if model exists
        if self.train_or_inference.inference is None and self.train_or_inference.model_prefix is not None:
            if os.path.isdir('./saved_model'):
                tmp_prefixes = glob.glob('./saved_model/*')
                prog = re.compile(r'./saved_model/(.+?)_')
                tmp = []
                for file_name in tmp_prefixes:
                    try:
                        prefix = prog.match(file_name).group(1)
                    except AttributeError:
                        pass
                    else:
                        tmp.append(prefix)
                tmp_prefixes = set(tmp)
                if self.train_or_inference.model_prefix in tmp_prefixes:
                    raise EnvironmentError('\n'
                                           'models with prefix {} already exists.\n'
                                           'Please specify a different prefix.'.format(self.train_or_inference.model_prefix)
                                           )

        config = tf.ConfigProto(allow_soft_placement=False, inter_op_parallelism_threads=1,
                   intra_op_parallelism_threads=1)

        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.config = config

        self.file_names = MiscFns.split_train_valid_and_test_files_fn()

        # in debug mode the numbers of recordings for training, test and validation are minimized for a debugging purpose
        if self.debug_mode:
            # for name in ('training', 'validation', 'test'):
            #     if name == 'training':
            #         del self.file_names[name][2:]
            #     else:
            #         del self.file_names[name][1:]
            self.file_names['training'] = self.file_names['training'][:2]
            self.file_names['validation'] = self.file_names['validation'][:1]
            self.file_names['test'] = self.file_names['test'][0:2]

            self.num_epochs = 3
            self.batches_per_epoch = 5
            self.gpu_id = 0

        # in inference mode, the numbers of recordings for training and validation are minimized
        if self.train_or_inference.inference is not None:
            for name in ('training', 'validation'):
                del self.file_names[name][1:]

        # the logarithmic filterbank
        self.log_filter_bank = MiscFns.log_filter_bank_fn()


# define nn models
class Model(object):
    def __init__(self, config, name):
        assert name in ('validation', 'training', 'test')
        self.name = name
        logging.debug('{} - model - initialize'.format(self.name))
        self.is_training = True if self.name == 'training' else False
        self.config = config

        if not self.is_training:
            self.reinitializable_iter_for_dataset = None
        self.batch = self._gen_batch_fn()  # generate mini-batch

        with tf.name_scope(self.name):
            with tf.variable_scope('full_conv', reuse=tf.AUTO_REUSE):
                logits_stereo = self._nn_model_fn()

            logits_stereo_flattened = flatten_maybe_padded_sequences(
                maybe_padded_sequences=logits_stereo,
                lengths=tf.tile(input=self.batch['num_frames'], multiples=[2]))
            logits_left_flattened, logits_right_flattened = tf.split(
                value=logits_stereo_flattened, num_or_size_splits=2, axis=0)

            logits_minor_flattened = tf.minimum(logits_left_flattened, logits_right_flattened)
            logits_larger_flattened = tf.maximum(logits_left_flattened, logits_right_flattened)

            labels_bool_flattened = flatten_maybe_padded_sequences(
                maybe_padded_sequences=self.batch['label'], lengths=self.batch['num_frames'])
            negated_labels_bool_flattened = tf.logical_not(labels_bool_flattened)
            labels_float_flattened = tf.cast(x=labels_bool_flattened, dtype=tf.float32)

            #When label is True, choose the smaller logits. Otherwise, choose the larger logits
            logits_mono_flattened = tf.where(
               tf.equal(labels_bool_flattened, True), logits_minor_flattened, logits_larger_flattened)

            #cross-entropy
            #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_float_flattened, logits=logits_mono_flattened)

            #weighted cross-entropy
            #A value `pos_weights > 1` decreases the false negative count, hence increasing the recall.
            #Conversely setting `pos_weights < 1` decreases the false positive count and increases the precision.
            loss = tf.nn.weighted_cross_entropy_with_logits(targets=labels_float_flattened, logits=logits_mono_flattened, pos_weight=1.1)

            #focal loss
            #loss = MiscFns.focal_loss(labels=labels_float_flattened, logits=logits_mono_flattened)

            loss = tf.reduce_mean(loss)

            if self.is_training:
                global_step = tf.train.get_or_create_global_step()
                learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step, \
                                                           self.config.batches_per_epoch * 7, 0.7, staircase=True)

                _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if _update_ops:
                    with tf.control_dependencies(_update_ops):
                        training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
                else:
                    training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

            pred_labels_flattened = tf.greater(logits_left_flattened+logits_right_flattened, 0)
            negated_pred_labels_flattened = tf.logical_not(pred_labels_flattened)

            # individual and ensemble statistics for test and validation
            if not self.is_training:
                with tf.name_scope('individual_and_ensemble_stats'):
                    with tf.variable_scope('{}_local_vars'.format(self.name), reuse=tf.AUTO_REUSE):
                        individual_tps_fps_tns_fns_var = tf.get_variable(
                            name='individual_tps_fps_tns_fns',
                            shape=[len(self.config.file_names[self.name]), 4],
                            dtype=tf.int32,
                            initializer=tf.zeros_initializer,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                        )

                        acc_loss_var = tf.get_variable(
                            name='acc_loss',
                            shape=[],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                        )

                        batch_counter_var = tf.get_variable(
                            name='batch_counter',
                            shape=[],
                            dtype=tf.int32,
                            initializer=tf.zeros_initializer,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                        )

                    loop_var_proto = collections.namedtuple(
                        'loop_var_proto',
                        ['sample_idx', 'batch_size', 'preds', 'negated_preds',
                         'labels', 'negated_labels', 'lengths', 'me_ids'])

                    def cond_fn(loop_var):
                        return tf.less(loop_var.sample_idx, loop_var.batch_size)

                    def body_fn(loop_var):
                        start_pos = tf.reduce_sum(loop_var.lengths[:loop_var.sample_idx])
                        end_pos = start_pos + loop_var.lengths[loop_var.sample_idx]
                        cur_preds = loop_var.preds
                        negated_cur_preds = loop_var.negated_preds
                        cur_labels = loop_var.labels
                        negated_cur_labels = loop_var.negated_labels
                        cur_preds, negated_cur_preds, cur_labels, negated_cur_labels = \
                            [value[start_pos:end_pos]
                             for value in [cur_preds, negated_cur_preds, cur_labels, negated_cur_labels]]
                        tps = tf.logical_and(cur_preds, cur_labels)
                        fps = tf.logical_and(cur_preds, negated_cur_labels)
                        tns = tf.logical_and(negated_cur_preds, negated_cur_labels)
                        fns = tf.logical_and(negated_cur_preds, cur_labels)
                        tps, fps, tns, fns = \
                            [tf.reduce_sum(tf.cast(value, tf.int32)) for value in [tps, fps, tns, fns]]
                        me_id = loop_var.me_ids[loop_var.sample_idx]
                        stats_var = individual_tps_fps_tns_fns_var
                        _new_value = stats_var[me_id] + tf.convert_to_tensor([tps, fps, tns, fns])
                        _update_stats = tf.scatter_update(
                            stats_var, me_id, _new_value, use_locking=True)
                        with tf.control_dependencies([_update_stats]):
                            sample_idx = loop_var.sample_idx + 1
                        loop_var = loop_var_proto(
                            sample_idx=sample_idx,
                            batch_size=loop_var.batch_size,
                            preds=loop_var.preds,
                            negated_preds=loop_var.negated_preds,
                            labels=loop_var.labels,
                            negated_labels=loop_var.negated_labels,
                            lengths=loop_var.lengths,
                            me_ids=loop_var.me_ids
                        )

                        return [loop_var]

                    sample_idx = tf.constant(0, dtype=tf.int32)
                    cur_batch_size = tf.shape(self.batch['num_frames'])[0]
                    loop_var = loop_var_proto(
                        sample_idx=sample_idx,
                        batch_size=cur_batch_size,
                        preds=pred_labels_flattened,
                        negated_preds=negated_pred_labels_flattened,
                        labels=labels_bool_flattened,
                        negated_labels=negated_labels_bool_flattened,
                        lengths=self.batch['num_frames'],
                        me_ids=self.batch['me_id']
                    )
                    final_sample_idx = tf.while_loop(
                        cond=cond_fn,
                        body=body_fn,
                        loop_vars=[loop_var],
                        parallel_iterations=self.config.batch_size,
                        back_prop=False,
                        return_same_structure=True
                    )[0].sample_idx

                    individual_tps_fps_tns_fns_float = tf.cast(individual_tps_fps_tns_fns_var, tf.float32)
                    tps, fps, _, fns = tf.unstack(individual_tps_fps_tns_fns_float, axis=1)
                    me_wise_precisions = tps / (tps + fps + 1e-7)
                    me_wise_recalls = tps / (tps + fns + 1e-7)
                    me_wise_f1s = 2. * me_wise_precisions * me_wise_recalls / \
                                  (me_wise_precisions + me_wise_recalls + 1e-7)
                    me_wise_prfs = tf.stack([me_wise_precisions, me_wise_recalls, me_wise_f1s], axis=1)
                    assert me_wise_prfs.shape.as_list() == [len(self.config.file_names[self.name]), 3]
                    average_me_wise_prf = tf.reduce_mean(me_wise_prfs, axis=0)
                    assert average_me_wise_prf.shape.as_list() == [3]

                    # ensemble stats
                    ensemble_tps_fps_tns_fns = tf.reduce_sum(individual_tps_fps_tns_fns_var, axis=0)
                    tps, fps, _, fns = tf.unstack(tf.cast(ensemble_tps_fps_tns_fns, tf.float32))
                    en_precision = tps / (tps + fps + 1e-7)
                    en_recall = tps / (tps + fns + 1e-7)
                    en_f1 = 2. * en_precision * en_recall / (en_precision + en_recall + 1e-7)
                    batch_counter_update_op = tf.assign_add(batch_counter_var, 1)
                    acc_loss_update_op = tf.assign_add(acc_loss_var, loss)
                    ensemble_prf_and_loss = tf.convert_to_tensor(
                        [en_precision, en_recall, en_f1, acc_loss_var / tf.cast(batch_counter_var, tf.float32)])

                    update_op_after_each_batch = tf.group(
                        final_sample_idx, batch_counter_update_op, acc_loss_update_op,
                        name='grouped update ops to be run after each batch'.replace(' ', '_'))
                    stats_after_each_epoch = dict(
                        individual_tps_fps_tns_fns=individual_tps_fps_tns_fns_var,
                        individual_prfs=me_wise_prfs,
                        ensemble_tps_fps_tns_fns=ensemble_tps_fps_tns_fns,
                        ensemble_prf_and_loss=ensemble_prf_and_loss,
                        average_prf=average_me_wise_prf
                    )

            '''
            # ensemble stats for training
            if self.is_training:
                with tf.name_scope('ensemble_stats'):
                    with tf.variable_scope('{}_local_vars'.format(self.name), reuse=tf.AUTO_REUSE):
                        ensemble_tps_fps_tns_fns_var = tf.get_variable(
                            name='ensemble_tps_fps_tns_fns',
                            shape=[4],
                            dtype=tf.int32,
                            initializer=tf.zeros_initializer,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                        )
                        acc_loss_var = tf.get_variable(
                            name='acc_loss',
                            shape=[],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                        )
                        batch_counter_var = tf.get_variable(
                            name='batch_counter',
                            shape=[],
                            dtype=tf.int32,
                            initializer=tf.zeros_initializer,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                        )

                    tps = tf.logical_and(pred_labels_flattened, labels_bool_flattened)
                    fps = tf.logical_and(pred_labels_flattened, negated_labels_bool_flattened)
                    tns = tf.logical_and(negated_pred_labels_flattened, negated_labels_bool_flattened)
                    fns = tf.logical_and(negated_pred_labels_flattened, labels_bool_flattened)
                    tps, fps, tns, fns = [tf.reduce_sum(tf.cast(value, tf.int32)) for value in [tps, fps, tns, fns]]

                    ensemble_tps_fps_tns_fns_update_op = tf.assign_add(
                        ensemble_tps_fps_tns_fns_var, tf.convert_to_tensor([tps, fps, tns, fns]))

                    acc_loss_update_op = tf.assign_add(acc_loss_var, loss)
                    batch_counter_update_op = tf.assign_add(batch_counter_var, 1)
                    ensemble_tps_fps_tns_fns_float = tf.cast(ensemble_tps_fps_tns_fns_var, tf.float32)
                    tps, fps, _, fns = tf.unstack(ensemble_tps_fps_tns_fns_float)
                    ensemble_precision = tps / (tps + fps + 1e-7)
                    ensemble_recall = tps / (tps + fns + 1e-7)
                    ensemble_f1 = 2. * ensemble_precision * ensemble_recall / \
                                  (ensemble_precision + ensemble_recall + 1e-7)
                    ensemble_loss = acc_loss_var / tf.cast(batch_counter_var, tf.float32)
                    ensemble_prf_and_loss = tf.convert_to_tensor(
                        [ensemble_precision, ensemble_recall, ensemble_f1, ensemble_loss])

                    update_op_after_each_batch = tf.group(
                        batch_counter_update_op, ensemble_tps_fps_tns_fns_update_op, acc_loss_update_op)
                    stats_after_each_epoch = dict(
                        ensemble_tps_fps_tns_fns=ensemble_tps_fps_tns_fns_var,
                        ensemble_prf_and_loss=ensemble_prf_and_loss
                    )

            '''


            # define tensorboard summaries
            with tf.name_scope('tensorboard_summary'):
                with tf.name_scope('statistics'):
                    if not self.is_training:
                        list_of_summaries = []
                        with tf.name_scope('ensemble'):
                            p, r, f, lo = tf.unstack(stats_after_each_epoch['ensemble_prf_and_loss'])
                            items_for_summary = dict(precision=p, recall=r, f1=f, average_loss=lo)
                            for item_name, item_value in items_for_summary.items():
                                tmp = tf.summary.scalar(item_name, item_value)
                                list_of_summaries.append(tmp)
                        with tf.name_scope('individual'):
                            p, r, f = tf.unstack(stats_after_each_epoch['average_prf'])
                            items_for_summary = dict(precision=p, recall=r, f1=f)
                            for item_name, item_value in items_for_summary.items():
                                tmp = tf.summary.scalar(item_name, item_value)
                                list_of_summaries.append(tmp)
                        statistical_summary = tf.summary.merge(list_of_summaries)
                    '''
                    else:
                        list_of_summaries = []
                        with tf.name_scope('ensemble'):
                            p, r, f, lo = tf.unstack(stats_after_each_epoch['ensemble_prf_and_loss'])
                            items_for_summary = dict(precision=p, recall=r, f1=f, average_loss=lo)
                            for item_name, item_value in items_for_summary.items():
                                tmp = tf.summary.scalar(item_name, item_value)
                                list_of_summaries.append(tmp)
                    statistical_summary = tf.summary.merge(list_of_summaries)           
                    '''

                with tf.name_scope('images'):
                    image_summary_length = int(6 * 16000 // 512)
                    labels_uint8 = self.batch['label'][:, :image_summary_length, :]
                    labels_uint8 = tf.cast(labels_uint8, tf.uint8) * 255
                    #assert labels_uint8.dtype == tf.uint8
                    labels_uint8 = labels_uint8[..., None]

                    _logits_left = tf.split(value=logits_stereo, num_or_size_splits=2, axis=0)[0]
                    
                    logits_prob_uint8 = tf.sigmoid(_logits_left[:, :image_summary_length, :])
                    logits_prob_uint8 = tf.cast(logits_prob_uint8 * 255., tf.uint8)
                    logits_prob_uint8 = logits_prob_uint8[..., None]

                    images = tf.concat([labels_uint8, logits_prob_uint8, tf.zeros_like(labels_uint8)], axis=-1)
                    images = tf.transpose(images, [0, 2, 1, 3])
                    images.set_shape([None, 88, image_summary_length, 3])
                    image_summary = tf.summary.image('images', images)

                if self.is_training:
                    with tf.name_scope('params'):
                        var_summary_dict = dict()
                        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                            var_summary_dict[var.op.name] = tf.summary.histogram(var.op.name, var)
                        param_summary = tf.summary.merge(list(var_summary_dict.values()))

        if self.is_training:
            op_dict = dict(
                training_op=training_op,
                #tb_summary=dict(statistics=statistical_summary, image=image_summary, parameter=param_summary),
                #tb_summary=dict(image=image_summary, parameter=param_summary),
                #update_op_after_each_batch=update_op_after_each_batch,
                #statistics_after_each_epoch=stats_after_each_epoch
            )
        else:
            op_dict = dict(
                tb_summary=dict(statistics=statistical_summary, image=image_summary),
                update_op_after_each_batch=update_op_after_each_batch,
                statistics_after_each_epoch=stats_after_each_epoch
            )

        self.op_dict = op_dict

    @staticmethod
    def parse(serialized):
        # Define a dict with the data-names and types we expect to find in the TFRecords file.
        features = {
            'spectrogram':
                tf.FixedLenFeature((), dtype=tf.string, default_value=''),
            'label':
                tf.FixedLenFeature((), dtype=tf.string, default_value=''),
            'h':
                tf.FixedLenFeature([1], dtype=tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
            'me_id':
                tf.FixedLenFeature([1], dtype=tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
            'shape_spec':
                tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
            'shape_label':
                tf.FixedLenFeature(shape=(2,), dtype=tf.int64)
        }

        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)

        # Get the image as raw bytes.
        spectrogram_raw = parsed_example['spectrogram']
        spectrogram = tf.decode_raw(spectrogram_raw, tf.float32)

        label_raw = parsed_example['label']
        label = tf.decode_raw(label_raw, tf.int32)

        h = parsed_example['h']
        me_id = parsed_example['me_id']

        h = tf.cast(h, dtype=tf.int32)
        me_id = tf.cast(me_id, dtype=tf.int32)

        shape_spec = parsed_example['shape_spec']
        shape_label = parsed_example['shape_label']

        spectrogram = tf.reshape(spectrogram, shape=shape_spec)

        label = tf.reshape(label, shape=shape_label)
        label = tf.cast(label, dtype=tf.bool)

        h = tf.reshape(h, shape=[])    #shape `[]` reshapes to a scalar
        me_id = tf.reshape(me_id, shape=[])

        out = {'spectrogram':spectrogram, 'label':label, 'num_frames':h, 'me_id':me_id}

        return out

    def _gen_batch_fn(self):
        #assert self.name in ['training', 'test']
        with tf.device('/cpu:0'):
            if self.name == 'training':
                dataset = tf.data.TFRecordDataset('/titan_data1/zhangwen/maps-tf/train/train.tfrecord')
            else:
                dataset = tf.data.TFRecordDataset('/titan_data1/zhangwen/maps-tf/test/test.tfrecord')

            dataset = dataset.map(Model.parse)
            dataset = dataset.shuffle(2)

            dataset = dataset.padded_batch(
                batch_size=self.config.batch_size,
                padded_shapes=dict(
                    spectrogram=[-1, 229, 2],
                    label=[-1, 88],
                    num_frames=[],
                    me_id=[]
                )
            )
            if self.is_training:
                dataset = dataset.repeat()

            if self.is_training:
                dataset_iter = dataset.make_one_shot_iterator()
                element = dataset_iter.get_next()
            else:
                reinitializabel_iter = dataset.make_initializable_iterator()
                self.reinitializable_iter_for_dataset = reinitializabel_iter
                element = reinitializabel_iter.get_next()

        return element

    def _nn_model_fn(self):
        inputs = self.batch['spectrogram']
        assert inputs.shape.as_list() == [None, None, 229, 2]

        # treat the two sound channels as independent examples
        # 在最后一个维度上将inputs分为3个tensor，然后将这3个tensor按第一个维度连接
        #inputs = tf.concat(tf.split(value=inputs, num_or_size_splits=3, axis=-1), axis=0)
        #inputs = tf.squeeze(inputs, axis=-1)    #删除最后一维，其维度为1

        #assert inputs.shape.as_list() == [None, None, 229]

        Net = MusicNet(training=self.is_training, name='Net')
        outputs = Net(spec_batch=inputs)

        return outputs


def main():
    warnings.simplefilter("ignore", ResourceWarning)
    MODEL_DICT = {}
    MODEL_DICT['config'] = Config()  # generate configurations

    # generate models
    #for name in ('training', 'validation', 'test'):
    for name in ('training', 'test'):
        MODEL_DICT[name] = Model(config=MODEL_DICT['config'], name=name)

    # placeholder for auxiliary information
    aug_info_pl = tf.placeholder(dtype=tf.string, name='aug_info_pl')
    aug_info_summary = tf.summary.text('aug_info_summary', aug_info_pl)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(MODEL_DICT['config'].gpu_id)

    with tf.Session(config=MODEL_DICT['config'].config) as sess:
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess, coord)
        # define model saver
        if MODEL_DICT['config'].train_or_inference.inference is not None or \
                MODEL_DICT['config'].train_or_inference.from_saved is not None or \
                MODEL_DICT['config'].train_or_inference.model_prefix is not None:
            MODEL_DICT['model_saver'] = tf.train.Saver(max_to_keep=200)

            logging.info('saved/restored variables:')
            for idx, var in enumerate(MODEL_DICT['model_saver']._var_list):
                logging.info('{}\t{}'.format(idx, var.op.name))

        # define summary writers
        summary_writer_dict = {}
        #for training_valid_or_test in ('training', 'validation', 'test'):
        for training_valid_or_test in ('training', 'test'):
            if training_valid_or_test == 'training':
                summary_writer_dict[training_valid_or_test] = tf.summary.FileWriter(
                    os.path.join(MODEL_DICT['config'].tb_dir, training_valid_or_test),
                    sess.graph
                )
            else:
                summary_writer_dict[training_valid_or_test] = tf.summary.FileWriter(
                    os.path.join(MODEL_DICT['config'].tb_dir, training_valid_or_test)
                )

        aug_info = []
        if MODEL_DICT['config'].train_or_inference.inference is not None:
            aug_info.append('inference with {}'.format(MODEL_DICT['config'].train_or_inference.inference))
            aug_info.append('inference with only the first 30 secs - {}'.format(MODEL_DICT['config'].test_with_30_secs))
        elif MODEL_DICT['config'].train_or_inference.from_saved is not None:
            aug_info.append('continue training from {}'.format(MODEL_DICT['config'].train_or_inference.from_saved))
        aug_info.append('learning rate - {}'.format(MODEL_DICT['config'].learning_rate))
        aug_info.append('tb dir - {}'.format(MODEL_DICT['config'].tb_dir))
        aug_info.append('debug mode - {}'.format(MODEL_DICT['config'].debug_mode))
        aug_info.append('batch size - {}'.format(MODEL_DICT['config'].batch_size))
        aug_info.append('num of batches per epoch - {}'.format(MODEL_DICT['config'].batches_per_epoch))
        aug_info.append('num of epochs - {}'.format(MODEL_DICT['config'].num_epochs))
        aug_info.append('training start time - {}'.format(datetime.datetime.now()))
        aug_info = '\n\n'.join(aug_info)
        logging.info(aug_info)
        summary_writer_dict['training'].add_summary(sess.run(aug_info_summary, feed_dict={aug_info_pl: aug_info}))

        logging.info('global vars -')
        for idx, var in enumerate(tf.global_variables()):
            logging.info("{}\t{}\t{}".format(idx, var.name, var.shape))

        logging.info('local vars -')
        for idx, var in enumerate(tf.local_variables()):
            logging.info('{}\t{}'.format(idx, var.name))

        #extract tf operations
        op_stat_summary_dict = {}
        #for training_valid_or_test in ('training', 'validation', 'test'):
        for training_valid_or_test in ('training', 'test'):
            op_list = []
            if training_valid_or_test == 'training':
                op_list.append(MODEL_DICT[training_valid_or_test].op_dict['training_op'])
                #op_list.append(MODEL_DICT[training_valid_or_test].op_dict['update_op_after_each_batch'])
                op_stat_summary_dict[training_valid_or_test] = dict(
                    op_list=op_list
                )

            else:
                op_list.append(MODEL_DICT[training_valid_or_test].op_dict['update_op_after_each_batch'])
                stat_op_dict = MODEL_DICT[training_valid_or_test].op_dict['statistics_after_each_epoch']
                tb_summary_dict = MODEL_DICT[training_valid_or_test].op_dict['tb_summary']
                op_stat_summary_dict[training_valid_or_test] = dict(
                    op_list=op_list,
                    stat_op_dict=stat_op_dict,
                    tb_summary_dict=tb_summary_dict
                )


        if MODEL_DICT['config'].train_or_inference.inference is not None:  # inference
            save_path = os.path.join('saved_model', MODEL_DICT['config'].train_or_inference.inference)
            print('save_path:{}'.format(save_path))
            MODEL_DICT['model_saver'].restore(sess, save_path)

            logging.info('do inference ...')
            # initialize local variables for storing statistics
            sess.run(tf.initializers.variables(tf.local_variables()))
            # initialize dataset iterator
            sess.run(MODEL_DICT['test'].reinitializable_iter_for_dataset.initializer)

            op_list = op_stat_summary_dict['test']['op_list']
            stat_op_dict = op_stat_summary_dict['test']['stat_op_dict']
            tb_summary_image = op_stat_summary_dict['test']['tb_summary_dict']['image']
            tb_summary_stats = op_stat_summary_dict['test']['tb_summary_dict']['statistics']

            batch_idx = 0
            op_list_with_image_summary = [tb_summary_image] + op_list
            logging.info('batch - {}'.format(batch_idx + 1))
            tmp = sess.run(op_list_with_image_summary)
            images = tmp[0]
            summary_writer_dict['test'].add_summary(images, 0)

            while True:
                try:
                    sess.run(op_list)
                except tf.errors.OutOfRangeError:
                    break
                else:
                    batch_idx += 1
                    logging.info('batch - {}'.format(batch_idx + 1))
            # write summary data
            summary_writer_dict[training_valid_or_test].add_summary(sess.run(tb_summary_stats), 0)

            # generate statistics
            stat_dict = sess.run(stat_op_dict)

            # display statistics
            MiscFns.display_stat_dict_fn(stat_dict)

        elif MODEL_DICT['config'].train_or_inference.from_saved is not None:  # restore saved model for training
            save_path = os.path.join('saved_model', MODEL_DICT['config'].train_or_inference.from_saved)
            MODEL_DICT['model_saver'].restore(sess, save_path)

            # reproduce statistics
            logging.info('reproduce results ...')
            sess.run(tf.initializers.variables(tf.local_variables()))
            #for valid_or_test in ('validation', 'test'):
            for valid_or_test in (['test']):
                sess.run(MODEL_DICT[valid_or_test].reinitializable_iter_for_dataset.initializer)
            #for valid_or_test in ('validation', 'test'):
            for valid_or_test in (['test']):
                logging.info(valid_or_test)

                op_list = op_stat_summary_dict[valid_or_test]['op_list']
                stat_op_dict = op_stat_summary_dict[valid_or_test]['stat_op_dict']
                statistical_summary = op_stat_summary_dict[valid_or_test]['tb_summary_dict']['statistics']
                image_summary = op_stat_summary_dict[valid_or_test]['tb_summary_dict']['image']

                batch_idx = 0
                op_list_with_image_summary = [image_summary] + op_list
                logging.info('batch - {}'.format(batch_idx + 1))
                tmp = sess.run(op_list_with_image_summary)
                images = tmp[0]
                summary_writer_dict[valid_or_test].add_summary(images, 0)

                while True:
                    try:
                        sess.run(op_list)
                    except tf.errors.OutOfRangeError:
                        break
                    else:
                        batch_idx += 1
                        logging.info('batch - {}'.format(batch_idx + 1))

                summary_writer_dict[valid_or_test].add_summary(sess.run(statistical_summary), 0)

                stat_dict = sess.run(stat_op_dict)

                MiscFns.display_stat_dict_fn(stat_dict)
        else:  # train from scratch and need to initialize global variables
            sess.run(tf.initializers.variables(tf.global_variables()))

        if MODEL_DICT['config'].train_or_inference.inference is None:
            for training_valid_test_epoch_idx in range(MODEL_DICT['config'].num_epochs):
                logging.info('\n\nepoch - {}/{}'.format(training_valid_test_epoch_idx + 1, MODEL_DICT['config'].num_epochs))

                sess.run(tf.initializers.variables(tf.local_variables()))

                #for valid_or_test in ('validation', 'test'):
                for valid_or_test in (['test']):
                    sess.run(MODEL_DICT[valid_or_test].reinitializable_iter_for_dataset.initializer)

                #for training_valid_or_test in ('training', 'validation', 'test'):
                for training_valid_or_test in ('training', 'test'):
                    logging.info(training_valid_or_test)

                    op_list = op_stat_summary_dict[training_valid_or_test]['op_list']
                    if training_valid_or_test == 'test':
                        stat_op_dict = op_stat_summary_dict[training_valid_or_test]['stat_op_dict']
                        statistical_summary = op_stat_summary_dict[training_valid_or_test]['tb_summary_dict']['statistics']
                        image_summary = op_stat_summary_dict[training_valid_or_test]['tb_summary_dict']['image']


                    if training_valid_or_test == 'training':
                        for batch_idx in range(MODEL_DICT['config'].batches_per_epoch):
                            if batch_idx % 1000 == 0:
                                print('batch_idx={}'.format(batch_idx))

                            sess.run(op_list)

                            #print('batch_idx={}'.format(batch_idx))
                            logging.debug('batch - {}/{}'.format(batch_idx + 1, MODEL_DICT['config'].batches_per_epoch))

                        '''
                        summary_writer_dict[training_valid_or_test].add_summary(
                            sess.run(image_summary), training_valid_test_epoch_idx + 1)
                        summary_writer_dict[training_valid_or_test].add_summary(
                            sess.run(statistical_summary), training_valid_test_epoch_idx + 1)
                        param_summary = MODEL_DICT[training_valid_or_test].op_dict['tb_summary']['parameter']
                        summary_writer_dict[training_valid_or_test].add_summary(
                            sess.run(param_summary), training_valid_test_epoch_idx + 1)

                        stat_dict = sess.run(stat_op_dict)
                        '''

                        if MODEL_DICT['config'].train_or_inference.model_prefix is not None:
                            save_path = MODEL_DICT['config'].train_or_inference.model_prefix + \
                                        '_' + 'epoch_{}_of_{}'.format(training_valid_test_epoch_idx + 1,
                                                                      MODEL_DICT['config'].num_epochs)
                            save_path = os.path.join('saved_model', save_path)
                            save_path = MODEL_DICT['model_saver'].save(
                                sess=sess,
                                save_path=save_path,
                                global_step=None,
                                write_meta_graph=False
                            )
                            logging.info('model saved to {}'.format(save_path))
                    else:
                        batch_idx = 0
                        op_list_with_image_summary = [image_summary] + op_list
                        logging.debug('batch - {}'.format(batch_idx + 1))
                        tmp = sess.run(op_list_with_image_summary)
                        images = tmp[0]
                        summary_writer_dict[training_valid_or_test].add_summary(
                            images,
                            training_valid_test_epoch_idx + 1
                        )

                        while True:
                            try:
                                sess.run(op_list)
                            except tf.errors.OutOfRangeError:
                                break
                            else:
                                batch_idx += 1
                                logging.debug('batch - {}'.format(batch_idx + 1))

                        summary_writer_dict[training_valid_or_test].add_summary(
                            sess.run(statistical_summary),
                            training_valid_test_epoch_idx + 1
                        )

                        stat_dict = sess.run(stat_op_dict)

                        MiscFns.display_stat_dict_fn(stat_dict)


        msg = 'training end time - {}'.format(datetime.datetime.now())
        logging.info(msg)
        summary_writer_dict['training'].add_summary(sess.run(aug_info_summary, feed_dict={aug_info_pl: msg}))

        #for training_valid_or_test in ('training', 'validation', 'test'):
        for training_valid_or_test in ('training', 'test'):
            summary_writer_dict[training_valid_or_test].close()


if __name__ == '__main__':
    main()
