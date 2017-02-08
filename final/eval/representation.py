'''
Created on Feb 8, 2017

@author: jonas
'''

import tensorflow as tf
from tensorflow.python.platform import gfile
import preprocessing.dataset
import models.sentiment
import numpy as np
import sqlite3
import pickle
from configparser import ConfigParser


class Representation:
    
    def __init__(self, cid=0, datafolder="/home/t2/data", databasefolder="/home/t2", max_companies=2, limit=25000, seqlength=30):
        self.dataset = preprocessing.dataset.DataSet(datafolder, databasefolder, max_companies)
        self.conn = sqlite3.connect(databasefolder + "/repr_" + str(cid) + ".db")
        self.initializeTables()
        self.limit = limit
        self.seqlength = seqlength
        self.cid = cid
        if not self.checkTable("vectors"):
            self.evaluateModel()
    
    def __del__(self):
        self.conn.close()
        
    def checkTable(self, name):
        c = self.conn.cursor()
        t = (name,)
        c.execute("SELECT * FROM `sqlite_master` WHERE `type`='table' AND `name`=?", t)
        if c.fetchone() is None:
            print("No table " + name + " found, initializing data.\n")
            return False
        else:
            c.execute("SELECT * FROM " + name)
            if c.fetchone() is None:
                print("No content in table " + name + " found, initializing data.\n")
                return False
        return True
    
    def initializeTables(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS `vectors` (`did` INTEGER NOT NULL PRIMARY KEY, `eval` BLOB)")
        
    def getData(self):
        c = self.conn.cursor()
        c.execute("SELECT * FROM `vectors`")
        result = c.fetchall()
        
        n = len(pickle.loads(result[0][1]))
        data = np.empty((len(result), n), dtype=np.float32)
        ids = []
        count = 0
        for r in result:
            ids.append(int(r[0]))
            tmp = pickle.loads(r[1])
            np.copyto(data[count], tmp)
            count += 1
        return {"data": data, "ids": ids}
        
    def evaluateModel(self):
        vocab_mapping = self.dataset.getVocabMapping()
        sequences = self.dataset.getDatasetFor(self.cid, self.seqlength, targets=False, dataids=True, limit=self.limit)
        seqlengths = np.full(len(sequences["data"]), self.seqlength, dtype=np.int32)
        targets = np.zeros((len(sequences["data"]), 16356), dtype=np.int32)
        c = self.conn.cursor()
        with tf.Session() as sess:
            model = self.load_model(sess, len(vocab_mapping))
            if model == None:
                return
            input_feed = {}
            input_feed[model.seq_input.name] = sequences["data"]
            input_feed[model.target.name] = targets
            input_feed[model.seq_lengths.name] = seqlengths
            output_feed = [model.representation]
            outputs = sess.run(output_feed, input_feed)
            count = 0
            for o in outputs[0][0]:
                t = (sequences["ids"][count], pickle.dumps(o, 0))
                c.execute("INSERT INTO `vectors` VALUES (?, ?)", t)
                count += 1
        self.conn.commit()
        
    def load_model(self, session, vocab_size):
        hyper_params = self.read_config_file()
        model = models.sentiment.SentimentModel(vocab_size,
                                                hyper_params["hidden_size"],
                                                1.0,
                                                hyper_params["num_layers"],
                                                hyper_params["grad_clip"],
                                                hyper_params["max_seq_length"],
                                                hyper_params["learning_rate"],
                                                hyper_params["lr_decay_factor"],
                                                25000,
                                                forward_only=True)
        ckpt = tf.train.get_checkpoint_state("data/checkpoints/")
        if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from {0}".format(ckpt.model_checkpoint_path))
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Double check you got the checkpoint_dir right...")
            print("Model not found...")
            model = None
        return model

    def read_config_file(self):
        '''
        Reads in config file, returns dictionary of network params
        '''
        config = ConfigParser()
        config.read("config.ini")
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
