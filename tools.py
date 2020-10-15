import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops
import tensorflow.nn as nn

import os
import glob
import re
import librosa
import librosa.display
import numpy as np
import logging

import madmom
import magenta.music

# contain the common, miscellaneous functions
class MiscFns(object):
    """Miscellaneous functions"""

    @staticmethod
    def filename_to_id(filename):
        """Translate a .wav or .mid path to a MAPS sequence id.
        This snippet is from [1]
        """
        return re.match(r'.*MUS-(.+)_[^_]+\.\w{3}',
                        os.path.basename(filename)).group(1)

    @staticmethod
    def log_filter_bank_fn():
        """
        generate a logarithmic filterbank
        """
        log_filter_bank_basis = madmom.audio.filters.LogarithmicFilterbank(
            bin_frequencies=librosa.fft_frequencies(sr=16000, n_fft=2048),
            num_bands=48,
            fmin=librosa.midi_to_hz([27])[0],
            fmax=librosa.midi_to_hz([114])[0] * 2. ** (1. / 48)
        )
        log_filter_bank_basis = np.array(log_filter_bank_basis)
        assert log_filter_bank_basis.shape[1] == 229
        assert np.abs(np.sum(log_filter_bank_basis[:, 0]) - 1.) < 1e-3
        assert np.abs(np.sum(log_filter_bank_basis[:, -1]) - 1.) < 1e-3

        return log_filter_bank_basis

    @staticmethod
    def spectrogram_fn(samples, log_filter_bank_basis, spec_stride):
        """
        generate spectrogram
        """
        num_frames = (len(samples) - 1) // spec_stride + 1
        stft = librosa.stft(y=samples, n_fft=2048, hop_length=spec_stride)
        assert num_frames <= stft.shape[1] <= num_frames + 1
        if stft.shape[1] == num_frames + 1:
            stft = stft[:, :num_frames]
        stft = stft / 1024
        stft = np.abs(stft)
        stft = 20 * np.log10(stft + 1e-7) + 140
        lm_mag = np.dot(stft.T, log_filter_bank_basis)

        # lm_mag.shape: (number of frame, 229)

        assert lm_mag.shape[1] == 229
        lm_mag = np.require(lm_mag, dtype=np.float32, requirements=['C'])

        return lm_mag

    @staticmethod
    def spectrogram_cqt(samples, sr, log_filter_bank_basis, spec_stride, norm_method='none'):
        "generate cqt"
        cqt = np.abs(librosa.cqt(y=samples, sr=sr, hop_length=spec_stride, fmin=39, n_bins=229, bins_per_octave=48))
        if norm_method == 'part_div':
            cqt = cqt / (np.max(np.abs(cqt)) + 1e-6)
        elif norm_method == 'part':
            cqt = cqt / (np.max(np.abs(cqt)) + 1e-6)
            cqt = (cqt - np.mean(cqt)) / (np.std(cqt) + 1e-6)
        else:
            raise NotImplementedError('un-know norm-operation: {}'.format(norm_method))

        cqt = np.dot(cqt.T, log_filter_bank_basis)

        assert cqt.shape[1] == 229
        cqt = np.require(cqt, dtype=np.float32, requirements=['C'])
        return cqt

    @staticmethod
    def spectrogram_mfcc(samples, sr, spec_stride, norm_method='none'):

        mel = librosa.feature.melspectrogram(
            samples,
            sr,
            hop_length=spec_stride,
            fmin=30.0,
            n_mels=229,
            htk=True
        )
        mel = librosa.power_to_db(mel)
        mfcc = librosa.feature.mfcc(S=mel, n_mfcc=229)
        if norm_method == 'part_div':
            mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-6)
        elif norm_method == 'part':
            mfcc = mfcc / (np.max(np.abs(mfcc)) + 1e-6)
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
        else:
            raise NotImplementedError('un-know norm-operation: {}'.format(norm_method))
        mfcc = mfcc.T
        assert mfcc.shape[1] == 229
        mfcc = np.require(mfcc, dtype=np.float32, requirements=['C'])
        return mfcc

    @staticmethod
    def times_to_frames_fn(spec_stride, start_time, end_time):
        """
        convert time to frame
        """
        assert spec_stride & 1 == 0
        start_sample = int(start_time * 16000)
        end_sample = int(end_time * 16000)
        start_frame = (start_sample + spec_stride // 2) // spec_stride
        end_frame = (end_sample + spec_stride // 2 - 1) // spec_stride
        return start_frame, end_frame + 1

    @staticmethod
    def label_fn(mid_file_name, num_frames, spec_stride):
        """labeling function"""
        label_matrix = np.zeros((num_frames, 88), dtype=np.bool_)
        note_seq = magenta.music.midi_file_to_sequence_proto(mid_file_name)
        note_seq = magenta.music.apply_sustain_control_changes(note_seq)
        for note in note_seq.notes:
            assert 21 <= note.pitch <= 108
            note_start_frame, note_end_frame = MiscFns.times_to_frames_fn(
                spec_stride=spec_stride,
                start_time=note.start_time,
                end_time=note.end_time
            )
            label_matrix[note_start_frame:note_end_frame, note.pitch - 21] = True

        return label_matrix

    @staticmethod
    def focal_loss(labels, logits, weight=None, alpha=0.5, gamma=2):
        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        Args:
         logits: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
         labels: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
         alpha: A scalar tensor for focal loss alpha hyper-parameter
         gamma: A scalar tensor for focal loss gamma hyper-parameter
        Returns:
            loss: A (scalar) tensor representing the value of the loss function
        """
        sigmoid_p = tf.nn.sigmoid(logits)
        #sigmoid_p = MiscFns.sigmoid(logits)
        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        pos_p_sub = array_ops.where(labels >= sigmoid_p, labels - sigmoid_p, zeros)
        neg_p_sub = array_ops.where(labels > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
        if weight is not None:
            weight = tf.constant(weight, dtype=per_entry_cross_ent.dtype)
            per_entry_cross_ent *= weight

        return per_entry_cross_ent

    @staticmethod
    def gen_stats(labels_bool_flattened, logits_bool_flattened, loss):
        """tf code for generating ensemble performance measures and mean loss"""
        assert labels_bool_flattened.dtype == tf.bool
        assert logits_bool_flattened.dtype == tf.bool

        stats_name_to_fn_dict = dict(
            tp=tf.metrics.true_positives,
            fn=tf.metrics.false_negatives,
            tn=tf.metrics.true_negatives,
            fp=tf.metrics.false_positives
        )
        kwargs = dict(
            labels=labels_bool_flattened,
            predictions=logits_bool_flattened
        )
        update_op_list = []
        value_dict = {}
        with tf.name_scope('statistics'):
            for stat_name, stat_fn in stats_name_to_fn_dict.items():
                value_op, update_op = stat_fn(name=stat_name, **kwargs)
                update_op_list.append(update_op)
                value_dict[stat_name] = value_op

            mean_loss_value_op, mean_loss_update_op = tf.metrics.mean(loss, name='average_loss')
            value_dict['average_loss'] = mean_loss_value_op
            update_op_list.append(mean_loss_update_op)

            merged_update_op = tf.group(update_op_list, name='merged_stat_update_op')

        stats_dict = dict(merged_update_op=merged_update_op)
        stats_dict['meta_data_dict'] = {}
        for meta_data_name in ('tp', 'fn', 'tn', 'fp'):
            stats_dict['meta_data_dict'][meta_data_name] = value_dict[meta_data_name]

        stats_dict['average_loss'] = value_dict['average_loss']

        tp = stats_dict['meta_data_dict']['tp']
        fn = stats_dict['meta_data_dict']['fn']
        fp = stats_dict['meta_data_dict']['fp']
        precision = stats_dict['precision'] = tp / (tp + fp + 1e-7)
        recall = stats_dict['recall'] = tp / (tp + fn + 1e-7)
        stats_dict['f1'] = 2. * precision * recall / (precision + recall + 1e-7)

        return stats_dict

    @staticmethod
    def split_train_valid_and_test_files_fn():
        """
        generate non-overlapped training-test set partition

        After downloading and unzipping the MAPS dataset,
        1. define an environment variable called maps to point to the directory of the MAPS dataset,
        2. populate test_dirs with the actual directories of the close and the ambient setting generated by
           the Disklavier piano,
        3. and populate train_dirs with the actual directoreis of the other 7 settings generated by the synthesizer.
        """
        '''test_dirs = ['ENSTDkCl_2/MUS', 'ENSTDkAm_2/MUS']
        train_dirs = ['AkPnBcht_2/MUS', 'AkPnBsdf_2/MUS', 'AkPnCGdD_2/MUS', 'AkPnStgb_2/MUS',
                      'SptkBGAm_2/MUS', 'SptkBGCl_2/MUS', 'StbgTGd2_2/MUS']
        maps_dir = os.environ['maps']
        '''
        test_dirs = ['test']
        train_dirs = ['train']
        maps_dir = '/titan_data1/zhangwen/maps/'

        test_files = []
        for directory in test_dirs:
            path = os.path.join(maps_dir, directory)
            path = os.path.join(path, '*.wav')
            wav_files = glob.glob(path)
            test_files += wav_files

        test_ids = set([MiscFns.filename_to_id(wav_file) for wav_file in test_files])
        assert len(test_ids) == 53

        training_files = []
        validation_files = []
        for directory in train_dirs:
            path = os.path.join(maps_dir, directory)
            path = os.path.join(path, '*.wav')
            wav_files = glob.glob(path)
            for wav_file in wav_files:
                me_id = MiscFns.filename_to_id(wav_file)
                if me_id not in test_ids:
                    training_files.append(wav_file)
                else:
                    validation_files.append(wav_file)

        assert len(training_files) == 139 and len(test_files) == 60 and len(validation_files) == 71

        return dict(training=training_files, test=test_files, validation=validation_files)

    @staticmethod
    def display_stat_dict_fn(stat_dict):
        """display statistics"""
        for stat_name, stat_value in stat_dict.items():
            if 'individual' in stat_name:
                logging.info(stat_name)
                for idx, sub_value in enumerate(stat_value):
                    logging.info('{} - {}'.format(idx, sub_value))
            else:
                logging.info(stat_name)
                logging.info(stat_value)


