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
    
    def __init__(self, datafolder="/home/t2/data", datafolderdatabasefolder="/home/t2", max_companies=2):
        self.datafolder = datafolder
        self.max_companies = max_companies
        if not os.path.isdir(databasefolder):
            os.mkdir(databasefolder)
        if not os.path.isdir(databasefolder):
            os.mkdir(databasefolder)
        self.first_company_index = -1
        self.conn = sqlite3.connect(databasefolder + "/data.db")
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
            print("No table " + name + " found, initializing data.\n")
            return False
        else:
            c.execute("SELECT * FROM " + name)
            if c.fetchone() is None:
                print("No content in table " + name + " found, initializing data.\n")
                return False
        c.execute("SELECT `name` FROM `companies` WHERE `num`=0")
        result = c.fetchone()
        t = (result[0],)
        c.execute("SELECT `wid` FROM `vocab` WHERE `word` = ?", t)
        result = c.fetchone()
        self.first_company_index = -1
        if result is not None:
            self.first_company_index = int(result[0]) 
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
        wdict = {0: 0, 1: 1, 2: 2}
        wnum = 3
        counter = 0
        print("Step 1: Creating temporary index mapping...\n")
        c.execute("SELECT `seqvec` FROM data")
        result = c.fetchall()
        for r in result:
            seqvec = pickle.loads(r[0])
            for w in seqvec:
                if w not in wdict and w > 2:
                    wdict[int(w)] = wnum
                    wnum += 1
            counter += 1
            if (counter % 10000) == 0:
                print("Checked " + str(counter) + " data lines.\n")
        print("Created temporary dictionary with " + str(len(wdict)) + " words.\n")
        print("Step 2: Cleaning vocab table...\n")
        c.execute("SELECT `wid`, `word` FROM `vocab`")
        result = c.fetchall()
        counter = 0
        c.execute("DELETE FROM `vocab`")
        for r in result:
            if int(r[0]) in wdict: 
                t = (str(wdict[int(r[0])]), r[1], )
                c.execute("INSERT INTO `vocab` VALUES (?, ?)", t)
                counter += 1
                if (counter % 10000) == 0:
                    print("Inserted " + str(counter) + " words.\n")
        print("Committing...\n")
        self.conn.commit()
        print("New vocab table completed.\n")
        print("Step 3: Refreshing sequence vectors...\n")
        c.execute("SELECT `id`, `seqvec` FROM `data`")
        result = c.fetchall()
        counter = 0
        for r in result:
            seqvec = pickle.loads(r[1])
            seqvec = list(map(lambda x: wdict[x], seqvec))
            seqvec = pickle.dumps(seqvec, 0)
            t = (seqvec, r[0], )
            c.execute("UPDATE `data` SET `seqvec` = ? WHERE `id` = ?", t)
            counter += 1
            if (counter % 10000) == 0:
                print("Refreshed " + str(counter) + " data lines.\n")
        print("Committing...\n")
        self.conn.commit()
        print("Dataset refreshed.\n")
        
    def removeCompanies(self, lower, higher):
        c = self.conn.cursor()
        t = (lower, higher,)
        c.execute("SELECT `basecid`, c FROM (SELECT `basecid`, COUNT(`id`) AS c FROM `data` GROUP BY `basecid`) WHERE c < ? OR c > ?", t)
        result = c.fetchall()
        print("Found " + str(len(result)) + " companies to delete.\n")
        for r in result:
            t = (r[0],)
            c.execute("DELETE FROM `companies` WHERE `cid` = ?", t)
            t = (r[0], r[0])
            c.execute("DELETE FROM `data` WHERE `basecid` = ? OR `targetcid` = ?", t)
        self.conn.commit()
        self.sanitizeCompanyNums()
    
    def sanitizeCompanyNums(self):
        c = self.conn.cursor()
        c.execute("SELECT * FROM `companies` ORDER BY `num` ASC")
        result = c.fetchall()
        c.execute("DELETE FROM `companies`")
        c.execute("VACUUM")
        for i in range(len(result)):
            t = (result[i][0], str(i), result[i][2],)
            c. execute("INSERT INTO `companies` VALUES (?, ?, ?)", t)
        self.conn.commit()
    
    def addCompaniesToVocab(self):
        c = self.conn.cursor()
        c.execute("SELECT max(`wid`) FROM `vocab`")
        result = c.fetchone()
        wnum = int(result[0]) + 1
        c.execute("SELECT `name` FROM `companies`")
        result = c.fetchall()
        self.first_company_index = wnum
        for r in result:
            t = (wnum, r[0],)
            c.execute("INSERT INTO `vocab` VALUES (?, ?)", t)
            wnum += 1
        self.conn.commit()
            
    
    def fillTables(self):
        print("Building data tables...")
        wnum = 3
        cnum = 0
        snum = 0
        lnum = 0
        wdict = dict()
        cset = set()
        stop = nltk.corpus.stopwords.words("german")
        c = self.conn.cursor()
        c.execute("INSERT INTO `vocab` VALUES (0, 'basecompanyplaceholder')")
        c.execute("INSERT INTO `vocab` VALUES (1, 'targetcompanyplaceholder')")
        c.execute("INSERT INTO `vocab` VALUES (2, '<PAD>')")
        wdict["basecompanyplaceholder"] = 0
        wdict["targetcompanyplaceholder"] = 1
        wdict["<PAD>"] = 2
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
                                    #tokens = [stemmer.stem(w) for w in tokens]
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
    
    def getVocabMapping(self):
        c = self.conn.cursor()
        c.execute("SELECT `wid`, `word` FROM `vocab`")
        result = c.fetchall()
        mapping = dict()
        for r in result:
            mapping[r[1]] = int(r[0])
        return mapping
        
    def getDatasetFor(self, basecompany, max_sequence_length, targets=True, companies=False, limit=-1, dataids=False):
        t = (basecompany,)
        c = self.conn.cursor()
        if not companies:
            c.execute("SELECT `name` FROM `companies` WHERE `cid`=?", t)
            result = c.fetchone()
            name = result[0]
            print("Selected basecompany " + name + ", loading data...\n")
            sql = "SELECT c.`num`, d.`seqvec`, d.`len`, d.`id` FROM `data` AS d JOIN `companies` AS c ON d.`targetcid` = c.`cid` WHERE `basecid`=?"
            if limit > 0:
                sql += " LIMIT " + str(limit)
            c.execute(sql, t)
        else:
            sql = "SELECT c.`num`, d.`seqvec`, d.`len`, d.`id` FROM `data` AS d JOIN `companies` AS c ON d.`targetcid` = c.`cid`"
            if limit > 0:
                sql += " LIMIT " + str(limit)
            c.execute(sql)
        result = c.fetchall()
        print("Found " + str(len(result)) + " sentences.\n")
        datalen = max_sequence_length
        if targets:
            datalen = max_sequence_length + 2
        if companies:
            datalen += 1
        didlist = []
        datalist = numpy.zeros(shape=(len(result), datalen), dtype=numpy.int32)
        count = 0
        for r in result:
            wordlist = pickle.loads(r[1])
            start = 0
            if companies:
                start = 1
                datalist[count][0] = self.first_company_index + r[0]
            for j in range(start, max_sequence_length):
                if j < len(wordlist):
                    datalist[count][j] = wordlist[j]
                else:
                    datalist[count][j] = 2
            if targets:
                datalist[count][max_sequence_length] = int(r[0])
                datalist[count][max_sequence_length + 1] = int(r[2])
            if dataids:
                didlist.append(int(r[3]))
            count += 1
        if dataids:
            return {"data": datalist, "ids": didlist}
        else:
            return datalist
        
