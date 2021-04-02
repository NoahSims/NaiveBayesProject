# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:59:07 2021

@author: Noah Sims
"""

# Import Module
import os
import string
import math
  
# Folder Paths
basePath = "C:/Users/Noah Sims/.spyder-py3/NaiveBayesProject/"
trainSpamPath = basePath + "assignment3_train/train/spam/"
trainHamPath = basePath + "assignment3_train/train/ham/"
testSpamPath = basePath + "assignment3_test/test/spam/"
testHamPath = basePath + "assignment3_test/test/ham/"
stopWordsFileName = "stopWords.txt"

numSpamFiles = 0
numHamFiles = 0

class VocabWord:
    def __init__(self, word, spam, ham):
        self.word = word.casefold()
        self.spam = spam
        self.ham = ham
        
    def compareToString(self, stg):
        if(self.word == stg.casefold()):
            return 0
        elif(self.word > stg.casefold()):
            return 1
        else:
            return -1
# end VocabWord
        

def createNewWord(word, isSpam):
    if isSpam:
        newWord = VocabWord(word, 1, 0)
    else:
        newWord = VocabWord(word, 0, 1)
    return newWord
#end createNewWorld()
            

def printVocabList(vList):
    os.chdir(basePath)
    outFile = open('output.txt', 'w')
    for i in vList:
        #print(i.word + ": Spam = " + str(i.spam) + "; Ham = " + str(i.ham))
        print(i.word + ": Spam = " + str(i.spam) + "; Ham = " + str(i.ham), file = outFile)
    outFile.close()
# end printVocabList()


def binarySearch(vList, word, start, end):
    if start == end:
        return start
            
    length = end - start
    middle = start + (int)(length / 2)
    #print("middle = " + str(middle) + ", len = " + str(len(vList)))
    compare = vList[middle].compareToString(word)
    if compare == 0:
        return middle
    elif compare < 0:
        return binarySearch(vList, word, middle + 1, end)
    elif compare > 0:
        return binarySearch(vList, word, start, middle)
#end binarySearch()
    

def insertVocab(vList, word, isSpam):
    if not vList:
        vList.append(createNewWord(word, isSpam))
        return vList
        
    index = binarySearch(vList, word, 0, len(vList) - 1)
    
    compare = vList[index].compareToString(word)
    if compare == 0:
        if isSpam:
            vList[index].spam += 1
        else:
            vList[index].ham += 1
    elif compare > 0:
        vList.insert(index, createNewWord(word, isSpam))
    elif compare < 0:
        if index == len(vList) - 1:
            vList.append(createNewWord(word, isSpam))
        else:
            vList.insert(index + 1, createNewWord(word, isSpam))

    return vList
# end insertVocab()


def populateVocabList(vList, path, isSpam, stopList):
    global numSpamFiles
    global numHamFiles
    # Change the directory
    os.chdir(path)
    
    # iterate through all files
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            if isSpam:
                numSpamFiles += 1
            else:
                numHamFiles += 1
            
            # read from file
            with open(file_path, encoding='utf8', errors='ignore') as f:
                #print(f)
                for line in f:
                    for word in line.split():
                        if word not in stopList:
                            vList = insertVocab(vList, word, isSpam)
            f.close()
    return vList
# end populateVocabList()

def classifySpam(vList, file, totalWordsHam, totalWordsSpam):
    pSpam = 0
    pHam = 0
    for line in file:
        for word in line.split():    
            index = binarySearch(vList, word, 0, len(vList) - 1)
            if vList[index].compareToString(word) == 0:
                #print(word)
                #print("spam = " + str(vList[index].spam) + "; pSpam = " + str(vList[index].spam / totalWordsSpam))
                #print("ham = " + str(vList[index].ham) + "; pHam = " + str(vList[index].ham / totalWordsHam))
                pSpam += math.log2((vList[index].spam + 1) / (totalWordsSpam + len(vList) - 1))
                pHam += math.log2((vList[index].ham + 1) / (totalWordsHam + len(vList) - 1))
    
    pSpam += math.log2(numSpamFiles / (numSpamFiles + numHamFiles))
    pHam += math.log2(numHamFiles / (numSpamFiles + numHamFiles))
    #print("pSpam = " + str(pSpam))
    #print("pHam = " + str(pHam))
    if pSpam > pHam:
        #print("This is spam")
        return True
    else:
        #print("100% certifies ham")
        return False
# end classify Spam()

def testDataSet(vList, path, isSpam, totalWordsHam, totalWordsSpam):
    totalFiles = 0
    correctTests = 0
    # Change the directory
    os.chdir(path)
    
    # iterate through all files
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            # read from file
            with open(file_path, encoding='utf8', errors='ignore') as f:
                #print(f)
                spamTest = classifySpam(vList, f, totalWordsHam, totalWordsSpam)
                totalFiles += 1
                if spamTest == isSpam:
                    correctTests += 1
            f.close()
            
    print("Total files = " + str(totalFiles))
    print("Correct tests = " + str(correctTests))
    print("Accuracy = " + str(correctTests / totalFiles))
    return correctTests / totalFiles
# end testDataSet()
                            

def populateStopWords(stopList, path, fileName):
    # Change the directory
    os.chdir(path)
    
    # Check whether file is in text format or not
    if fileName.endswith(".txt"):
        file_path = f"{path}\{fileName}"
        # read from file
        with open(file_path, encoding='utf8', errors='ignore') as f:
            for line in f:
                for word in line.split():
                    stopList.append(word) 
    return stopList
# end populateStopWords()

# -- MAIN --
vocabList = []
stopList = []

print("Reading training data")
# train spam
vocabList = populateVocabList(vocabList, trainSpamPath, True, stopList)
# train ham
vocabList = populateVocabList(vocabList, trainHamPath, False, stopList)
#printVocabList(vocabList)
print("Done")

print("Testing learned data")
# count spam and ham words
totalWordsSpam = 0
totalWordsHam = 0
for i in vocabList:
    totalWordsSpam += i.spam
    totalWordsHam += i.ham

# test spam
print("Spam Test")
testDataSet(vocabList, testSpamPath, True, totalWordsHam, totalWordsSpam)
# test ham
print("Ham test")
testDataSet(vocabList, testHamPath, False, totalWordsHam, totalWordsSpam)
print("Done")


# without stop words
print("\n")
# get stop words
print("Reading stop words file")
stopList = populateStopWords(stopList, basePath, stopWordsFileName)
vocabList = []

print("Reading training data")
# train spam
vocabList = populateVocabList(vocabList, trainSpamPath, True, stopList)
# train ham
vocabList = populateVocabList(vocabList, trainHamPath, False, stopList)
#printVocabList(vocabList)
print("Done")

print("Testing learned data")
# count spam and ham words
totalWordsSpam = 0
totalWordsHam = 0
for i in vocabList:
    totalWordsSpam += i.spam
    totalWordsHam += i.ham
    
# test spam
print("Spam Test")
testDataSet(vocabList, testSpamPath, True, totalWordsHam, totalWordsSpam)
# test ham
print("Ham test")
testDataSet(vocabList, testHamPath, False, totalWordsHam, totalWordsSpam)
print("Done")