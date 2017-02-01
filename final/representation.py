'''
Created on Jan 31, 2017

@author: jonas
'''

import tensorflow as tf
from tensorflow.python.platform import gfile
from preprocessing.dataset import DataSet
import models.sentiment
import numpy as np
import sqlite3
import pickle
import ConfigParser

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('config_file', 'config.ini', 'Path to configuration file.')
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/', 'Directory to store/restore checkpoints')

def main():
    s = DataSet("/home/jonas/kit/mlgpu/data/", "/home/jonas/kit/mlgpu/")
    vocab_mapping = s.getVocabMapping()
    dataset = s.getDatasetFor(80354, 30, targets=False, dataids=True, limit=25000)
    seqlengths = np.full((len(dataset["data"]), ), 30)
    targets = np.zeros((len(dataset["data"]), 16356))
    conn = sqlite3.connect("/home/jonas/kit/mlgpu/repr_80354.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS `vectors` (`did` INTEGER NOT NULL PRIMARY KEY, `repr` BLOB)")
    with tf.Session() as sess:
        model = load_model(sess, len(vocab_mapping))
        if model == None:
            return
        input_feed = {}
        input_feed[model.seq_input.name] = dataset["data"]
        input_feed[model.target.name] = targets
        input_feed[model.seq_lengths.name] = seqlengths
        output_feed = [model.representation]
        outputs = sess.run(output_feed, input_feed)
        count = 0
        for o in outputs:
            t = (str(dataset["ids"][count]), pickle.dumps(o.tolist(), 0))
            c.execute("INSERT INTO `vectors` (?, ?)")
        conn.commit()

def load_model(session, vocab_size):
    hyper_params = read_config_file()
    model = models.sentiment.SentimentModel(vocab_size,
        hyper_params["hidden_size"],
        1.0,
        hyper_params["num_layers"],
        hyper_params["grad_clip"],
        hyper_params["max_seq_length"],
        hyper_params["learning_rate"],
        hyper_params["lr_decay_factor"],
        1)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print "Reading model parameters from {0}".format(ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print "Double check you got the checkpoint_dir right..."
        print "Model not found..."
        model = None
    return model

def read_config_file():
    '''
    Reads in config file, returns dictionary of network params
    '''
    config = ConfigParser.ConfigParser()
    config.read(FLAGS.config_file)
    dic = {}
    sentiment_section = "sentiment_network_params"
    general_section = "general"
    dic["num_layers"] = config.getint(sentiment_section, "num_layers")
    dic["hidden_size"] = config.getint(sentiment_section, "hidden_size")
    dic["dropout"] = config.getfloat(sentiment_section, "dropout")
    dic["batch_size"] = config.getint(sentiment_section, "batch_size")
    dic["train_frac"] = config.getfloat(sentiment_section, "train_frac")
    dic["learning_rate"] = config.getfloat(sentiment_section, "learning_rate")
    dic["lr_decay_factor"] = config.getfloat(sentiment_section, "lr_decay_factor")
    dic["grad_clip"] = config.getint(sentiment_section, "grad_clip")
    dic["use_config_file_if_checkpoint_exists"] = config.getboolean(general_section,
        "use_config_file_if_checkpoint_exists")
    dic["max_epoch"] = config.getint(sentiment_section, "max_epoch")
    dic ["max_vocab_size"] = config.getint(sentiment_section, "max_vocab_size")
    dic["max_seq_length"] = config.getint(general_section,
        "max_seq_length")
    dic["steps_per_checkpoint"] = config.getint(general_section,
        "steps_per_checkpoint")
    return dic

if __name__ == "__main__":
    main()
    