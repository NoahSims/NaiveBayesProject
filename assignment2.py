# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:59:07 2021

@author: Noah Sims
"""

# Import Module
import os
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

# an object containing a word and ints to track how many times word appears in spam and ham files
class VocabWord:
    def __init__(self, word, spam, ham):
        self.word = word.casefold()
        self.spam = spam
        self.ham = ham
        
    # compares this VocabWord to a string
    def compareToString(self, stg):
        if(self.word == stg.casefold()):
            return 0
        elif(self.word > stg.casefold()):
            return 1
        else:
            return -1
    # end compareToString()
# end VocabWord
        
# instantiates a VocabWord as either spam or ham, depending on the value of isSpam. Returns VocabWord
def createNewWord(word, isSpam):
    if isSpam:
        newWord = VocabWord(word, 1, 0)
    else:
        newWord = VocabWord(word, 0, 1)
    return newWord
#end createNewWorld()            

# preforms binary search of vocabList, returning the index of word, or index where word should go if not in list
def binarySearch(vList, word, start, end):
    if start == end:
        return start
            
    length = end - start
    middle = start + (int)(length / 2)
    compare = vList[middle].compareToString(word)
    
    if compare == 0:
        return middle
    elif compare < 0:
        return binarySearch(vList, word, middle + 1, end)
    elif compare > 0:
        return binarySearch(vList, word, start, middle)
#end binarySearch()
    
# uses binarySearch to find word in vocabList. If word exists adds 1 to either spam or ham. If word doesn't exist,
# inserts word at index. Returns updated vocabList
def insertVocab(vList, word, isSpam):
    # if empty list
    if not vList:
        vList.append(createNewWord(word, isSpam))
        return vList
        
    # get index
    index = binarySearch(vList, word, 0, len(vList) - 1)
    
    compare = vList[index].compareToString(word)
    if compare == 0: # if word is already in vList
        if isSpam:
            vList[index].spam += 1
        else:
            vList[index].ham += 1
    # if word not in list, determine which side of index it belongs on to maintain sorted order
    elif compare > 0:
        vList.insert(index, createNewWord(word, isSpam))
    elif compare < 0:
        if index == len(vList) - 1:
            vList.append(createNewWord(word, isSpam))
        else:
            vList.insert(index + 1, createNewWord(word, isSpam))

    return vList
# end insertVocab()

# reads all files in path, inserting each word to vocabList, either as a new word, or incrementing existing words.
# stopList contains a list of words to be excluded from the vocabList
# Returns vocabList
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
            # Count total number of files, for use when classifying spam
            if isSpam:
                numSpamFiles += 1
            else:
                numHamFiles += 1
            
            # read from file
            with open(file_path, encoding='utf8', errors='ignore') as f:
                for line in f:
                    for word in line.split():
                        if word not in stopList:
                            vList = insertVocab(vList, word, isSpam)
            f.close()
    return vList
# end populateVocabList()

# takes in a single text file and classifies it as either spam or not spam based on the values in vocabList
# Returns true if determined as spam, false if determined not spam
def classifySpam(vList, file, totalWordsHam, totalWordsSpam):
    pSpam = 0
    pHam = 0
    for line in file:
        for word in line.split():    
            # find word in vocabList
            index = binarySearch(vList, word, 0, len(vList) - 1)
            # if word is in vocabList
            if vList[index].compareToString(word) == 0:
                pSpam += math.log2((vList[index].spam + 1) / (totalWordsSpam + len(vList) - 1))
                pHam += math.log2((vList[index].ham + 1) / (totalWordsHam + len(vList) - 1))
    
    # Naive Bayes Algorithm
    pSpam += math.log2(numSpamFiles / (numSpamFiles + numHamFiles))
    pHam += math.log2(numHamFiles / (numSpamFiles + numHamFiles))
    if pSpam > pHam:
        return True
    else:
        return False
# end classify Spam()

# reads all files in path, then uses classifySpam to determine if each file is spam or not spam
# returns the accuracy of classifySpam results as a float value
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
                # classifySpam
                spamTest = classifySpam(vList, f, totalWordsHam, totalWordsSpam)
                totalFiles += 1
                # if classifySpam was correct
                if spamTest == isSpam:
                    correctTests += 1
            f.close()
            
    print("Total files = " + str(totalFiles))
    print("Correct tests = " + str(correctTests))
    print("Accuracy = " + str(correctTests / totalFiles))
    return correctTests / totalFiles
# end testDataSet()
                            
# reads file from path, creating a list from the words contained within. Returns the list of words
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

# PART ONE - NOT USING STOP WORDS
print("Reading training data")
# in this section, stopList is empty, so no words are excluded from vocabList
# train spam
vocabList = populateVocabList(vocabList, trainSpamPath, True, stopList)
# train ham
vocabList = populateVocabList(vocabList, trainHamPath, False, stopList)

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
print("\n")


# PART TWO - USING STOP WORDS
# get stop words
print("Reading stop words file")
stopList = populateStopWords(stopList, basePath, stopWordsFileName)
vocabList = []

print("Reading training data")
# train spam
vocabList = populateVocabList(vocabList, trainSpamPath, True, stopList)
# train ham
vocabList = populateVocabList(vocabList, trainHamPath, False, stopList)

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