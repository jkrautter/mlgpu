'''
Created on Dec 13, 2016

@author: jonas
'''

import sqlite3
import os
import glob
import string
import nltk
import pickle
import math
import numpy

class DataSet:
    
    def __init__(self, datafolder="../../data/", databasefolder="../../", max_companies=2):
        self.datafolder = datafolder
        self.max_companies = max_companies
        if not os.path.isdir(databasefolder):
            os.mkdir(databasefolder)
        if not os.path.isdir(databasefolder):
            os.mkdir(databasefolder)
        self.conn = sqlite3.connect(databasefolder + "data.db")
        self.initializeTables()
        if not self.checkTable("companies") or not self.checkTable("vocab") or not self.checkTable("data"):
            self.fillTables()
        else:
            print("Data successfully loaded.\n")
    
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
        c.execute("CREATE TABLE IF NOT EXISTS `companies` (`cid` INTEGER NOT NULL PRIMARY KEY, `num` INTEGER NOT NULL, `name` TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS `vocab` (`wid` INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, `word` TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS `data` (`id` INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, " 
                  + "`basecid` INTEGER NOT NULL, `targetcid` INTEGER NOT NULL, `seqvec` BLOB, `len` INTEGER,"
                  + "FOREIGN KEY(`basecid`) REFERENCES `companies`(`cid`),"
                  + "FOREIGN KEY(`targetcid`) REFERENCES `companies`(`cid`))")
    
    def findNextCompany(self, s, start = 0):
        tagstart = s.find("<eb-comp", start)
        if tagstart != -1:
            idstart = s.find("id='", tagstart) + 4
            namestart = s.find(">", tagstart) + 1 
            cid = int(s[idstart:s.find("'", idstart)].split(",")[0])
            endstart = s.find("</", namestart)
            if endstart == -1:
                return -1
            name = s[namestart:endstart]
            return {"id": cid, "name": name, "start": tagstart, "end": endstart + 10}
        return -1

    def sanitizeVocab(self):
        c = self.conn.cursor()
        wdict = set()
        c.execute("SELECT `seqvec` FROM data")
        result = c.fetchall()
        for r in result:
            wdict.add(int(r[0]))
        c.execute("SELECT `wid` FROM `vocab`")
        result = c.fetchall()
        for r in result:
            if int(r[0]) not in wdict:
                t = (int(r[0]),)
                c.execute("DELETE FROM `vocab` WHERE `wid` = ?", t)
        self.conn.commit()        
        
    def removeCompanies(self, lower, higher):
        c = self.conn.cursor()
        c.execute("SELECT `cid` FROM `companies`")
        result = c.fetchall()
        for r in result:
            t = (r[0],)
            c.execute("SELECT count(`id`) FROM `data` WHERE `basecid` = ?")
            countres = c.fetchone()
            if int(countres[0]) > higher or int(countres[0]) < lower:
                c.execute("DELETE FROM `companies` WHERE `cid` = ?", t)
                t = (r[0], r[0])
                c.execute("DELETE FROM ` data` WHERE `basecid` = ? OR `targetid` = ?", t)
            self.conn.commit()
        self.sanitizeVocab()
    
    def fillTables(self):
        print("Building data tables...")
        wnum = 2
        cnum = 0
        snum = 0
        lnum = 0
        wdict = dict()
        cset = set()
        stop = nltk.corpus.stopwords.words("german")
        c = self.conn.cursor()
        c.execute("INSERT INTO `vocab` VALUES (0, 'basecompanyplaceholder')")
        c.execute("INSERT INTO `vocab` VALUES (1, 'targetcompanyplaceholder')")
        wdict["basecompanyplaceholder"] = 0
        wdict["targetcompanyplaceholder"] = 1
        self.conn.commit()
        for f in glob.glob(self.datafolder + "*.txt"):
            print("Processing file " + f + "...\n")
            datafile = open(f)   
            for line in datafile:
                lnum += 1
                if lnum % 10000 == 0:
                    print("Processing line: " + str(lnum) + "\n")
                if not "http" in line:
                    companies = []
                    nextcomp = self.findNextCompany(line, 0)
                    tempcnum = 0
                    while nextcomp != -1:
                        if nextcomp["id"] not in cset and "\n" not in nextcomp["name"]:
                            t = (nextcomp["id"], cnum, nextcomp["name"])
                            cset.add(nextcomp["id"])
                            c.execute("INSERT INTO `companies` VALUES (?, ?, ?)", t)
                            cnum += 1
                            if cnum % 10000 == 0:
                                print("Found " + str(cnum) + " companies, continuing...\n")
                        companies.append(nextcomp)
                        line = line[:nextcomp["start"]] + "[<[" + str(tempcnum) + "]>]" + line[nextcomp["end"]:]
                        tempcnum += 1
                        nextcomp = self.findNextCompany(line, nextcomp["start"] + 6 + math.ceil(math.log10(tempcnum)))
                    if len(companies) <= self.max_companies:
                        snum += 1
                        if snum % 10000 == 0:
                            print("Lines processed for database: " + str(snum) + "\n")
                            self.conn.commit()
                        for i in range(len(companies)):
                            baseline = line.replace("[<[" + str(i) + "]>]", " basecompanyplaceholder ")
                            for j in range(len(companies)):
                                if i != j:
                                    targetline = baseline.replace("[<[" + str(j) + "]>]", " targetcompanyplaceholder ")
                                    for k in range(len(companies)):
                                        if k != j and k != i:
                                            targetline = targetline.replace("[<[" + str(k) + "]>]", companies[k]["name"])
                                    targetline = "".join([ch for ch in targetline if ch not in string.punctuation and ch not in string.digits])
                                    targetline = targetline.lower()
                                    tokens = nltk.word_tokenize(targetline, "german")
                                    stemmer = nltk.stem.snowball.GermanStemmer()
                                    tokens = [stemmer.stem(w) for w in tokens]
                                    #tokens = [w for w in tokens if w not in stop]
                                    for w in tokens:
                                        if w not in wdict:
                                            wdict[w] = wnum
                                            t = (wnum, w,) 
                                            wnum += 1
                                            c.execute("INSERT INTO `vocab` (`wid`, `word`) VALUES (?, ?)", t)
                                            if wnum % 10000 == 0:
                                                print("Found " + str(wnum) + " words, continuing...\n")                
                                    seqvec = list(map(lambda x: wdict[x], tokens))
                                    t = (companies[i]["id"], companies[j]["id"], pickle.dumps(seqvec, 0), len(seqvec),)
                                    c.execute("INSERT INTO `data` (`basecid`, `targetcid`, `seqvec`, `len`) VALUES (?, ?, ?, ?)", t)
            self.conn.commit()
            datafile.close()
        print("All files processed, " + str(wnum) + " words, " + str(cnum) + " companies and " + str(snum) + " sentences found.")
        
    def getStringRepresentationFor(self, did):
        t = (did,)
        c = self.conn.cursor()
        c.execute("SELECT `seqvec` FROM `data` WHERE `id` = ?", t)
        result = c.fetchone()
        seqvec = pickle.loads(result[0])
        words = []
        for wid in seqvec:
            t = (wid,)
            c.execute("SELECT `word` FROM `vocab` WHERE `wid` = ?", t)
            result = c.fetchone()
            words.append(result[0])
        return str(words)
    
    def getVocabSize(self):
        c = self.conn.cursor()
        c.execute("SELECT count(`wid`) FROM `vocab`")
        result = c.fetchone()
        return int(result[0])
        
    def getDatasetFor(self, basecompany, max_sequence_length):
        t = (basecompany,)
        c = self.conn.cursor()
        c.execute("SELECT `name` FROM `companies` WHERE `cid`=?", t)
        result = c.fetchone()
        name = result[0]
        print("Selected basecompany " + name + ", loading data...\n")
        c.execute("SELECT c.`num`, d.`seqvec`, d.`len` FROM `data` AS d JOIN `companies` AS c ON d.`targetcid` = c.`cid` WHERE `basecid`=?", t)
        result = c.fetchall()
        print("Found " + str(len(result)) + " sentences including this basecompany.\n")
        c.execute("SELECT COUNT(`wid`) FROM `vocab`")
        tempresult = c.fetchone()
        numwords = int(tempresult[0])
        datalist = numpy.zeros(shape=(len(result), max_sequence_length + 2), dtype=numpy.int32)
        count = 0
        for r in result:
            wordlist = pickle.loads(r[1])
            for j in range(max_sequence_length):
                if j < len(wordlist):
                    datalist[count][j] = wordlist[j]
                else:
                    datalist[count][j] = numwords + 1
                    
            datalist[count][max_sequence_length] = r[0]
            datalist[count][max_sequence_length + 1] = r[2]
        return datalist
        