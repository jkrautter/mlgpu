'''
Created on Dec 13, 2016

@author: jonas
'''

import sqlite3
import os
import glob
import string
import nltk

class DataSet:
    
    def __init__(self, datafolder="../../data/", databasefolder="../../"):
        self.datafolder = datafolder
        if not os.path.isdir(databasefolder):
            os.mkdir(databasefolder)
        if not os.path.isdir(databasefolder):
            os.mkdir(databasefolder)
        self.conn = sqlite3.connect(databasefolder + "data.db")
        self.initializeTables()
        if not self.checkTable("companies"):
            self.fillCompanyTable()
        else:
            print("Company data successfully loaded.\n")
        if not self.checkTable("vocab"):
            self.fillVocabTable()
        else:
            print("Vocabulary data successfully loaded.\n")
    
    def __del__(self):
        self.conn.close()
    
    def checkTable(self, name):
        c = self.conn.cursor()
        t = (name,)
        c.execute("SELECT * FROM `sqlite_master` WHERE `type`='table' AND `name`=?", t)
        if c.fetchone() is None:
            return False
        else:
            c.execute("SELECT * FROM " + name)
            if c.fetchone() is None:
                return False
        return True
    
    def initializeTables(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS `companies` (`cid` INTEGER NOT NULL PRIMARY KEY, `name` TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS `vocab` (`wid` INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, `word` TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS `data` (`id` INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, `seqvec` BLOB, `len` INT)")
    
    def findNextCompany(self, s, start = 0):
        tagstart = s.find("<eb-comp", start)
        if tagstart != -1:
            idstart = s.find("id='", tagstart) + 4
            namestart = s.find(">", tagstart) + 1 
            cid = int(s[idstart:s.find("'", idstart)].split(",")[0])
            name = s[namestart:s.find("</", namestart)]
            return {"id": cid, "name": name, "start": namestart}
        return -1
    
    def fillCompanyTable(self):
        print("Finding companies...\n")
        cset = set()
        num = 0
        c = self.conn.cursor()
        for f in glob.glob(self.datafolder + "*.txt"):
            print("Processing file " + f + "...\n")
            datafile = open(f)
            content = datafile.read()
            nextcomp = self.findNextCompany(content, 0)
            while nextcomp != -1:
                if nextcomp["id"] not in cset and "\n" not in nextcomp["name"]:
                    num += 1
                    t = (nextcomp["id"], nextcomp["name"])
                    cset.add(nextcomp["id"])
                    c.execute("INSERT INTO `companies` VALUES (?, ?)", t)
                    if num % 10000 == 0:
                        print("Found " + str(num) + " companies, continuing...\n")
                nextcomp = self.findNextCompany(content, nextcomp["start"])
            datafile.close()
            self.conn.commit()
        print("All files processed, " + str(num) + " companies found.")
    
    def fillVocabTable(self):
        print("Building vocabulary...")
        num = 0
        wset = set()
        stop = nltk.corpus.stopwords.words("german")
        c = self.conn.cursor()
        for f in glob.glob(self.datafolder + "*.txt"):
            print("Processing file " + f + "...\n")
            datafile = open(f)
            content = datafile.read()
            print("Removing punctuation and digits...\n")
            content = "".join([ch for ch in content if ch not in string.punctuation and ch not in string.digits])
            print("Tokenizing...\n")
            content = nltk.word_tokenize(content, "german")
            print("Processing words...")
            for w in content:
                if w not in wset and w not in stop:
                    wset.add(w)
                    num += 1
                    t = (w,)
                    c.execute("INSERT INTO `vocab` (`word`) VALUES (?)", t)
                    if num % 10000 == 0:
                        print("Found " + str(num) + " words, continuing...\n")
            self.conn.commit()
            datafile.close()
        print("All files processed, " + str(num) + " words found.")
        
        