import json
import math
import statistics
import numpy as np
import sys

# Set the number of features available
numFeatures = 0

# Set the data set directory path
dataSetPath = '.'
trainSetName = '/data/train.csv.home.data'
testSetName = '/data/test.csv.home.data'
validateSetPattern = '/data/train%d.csv.home.data'

# The following are global dictionaries used to store the data from various sets (test, train, dev, cross-validate)
trainLabelsDictionary = {}
trainFeaturesDictionary = {}
validateLabelsDictionary = {}
validateFeaturesDictionary = {}
testLabelsDictionary = {}
testFeaturesDictionary = {}

def main(path):
    global trainLabelsDictionary
    global testLabelsDictionary
    global trainFeaturesDictionary
    global testFeaturesDictionary
    global numFeatures
    global dataSetPath

    dataSetPath = path

    trainLabelsDictionary, trainFeaturesDictionary = readTrainFile()
    testLabelsDictionary, testFeaturesDictionary = readTestFile()
    print(testLabelsDictionary)

    features = []
    for featureVector in trainFeaturesDictionary.values():
        features += list(featureVector.keys())
    for featureVector in testFeaturesDictionary.values():
        features += list(featureVector.keys())
    numFeatures = max(features)

    print('\n---------------THESE ARE THE STATISTICS FOR THE DATA SETS---------------')

    print('The size of the training set is %d' % (len(trainLabelsDictionary)))
    print('The size of the test set is %d' % (len(testLabelsDictionary)))

    print('\n---------------THESE ARE THE RESULTS FOR THE BAGGED FORESTS---------------')

    features = [x for x in range(1,numFeatures+1)]
    trees = []
    for i in range(0,100):
        initialDataSet = np.random.choice(len(trainLabelsDictionary), len(trainLabelsDictionary), replace=True)
        tree = id3Algorithm(initialDataSet, features, None, 0, 3)
        trees.append(tree)
        print('Tree number %d completed' % (i+1))

    trainAccuracy, transformedTrainData = evaluate_batch(trainLabelsDictionary, trainFeaturesDictionary, trees)
    testAccuracy, transformedTestData = evaluate_batch(testLabelsDictionary, testFeaturesDictionary, trees)
    print('Train Accuracy = %f Percent\n' % (trainAccuracy))
    print('Test Accuracy = %f Percent\n' % (testAccuracy))
    with open('data/trees/train.features.data', 'w') as f:
        json.dump(transformedTrainData, f)
    with open('data/trees/train.labels.data', 'w') as f:
        json.dump(trainLabelsDictionary, f)
    with open('data/trees/test.features.data', 'w') as f:
        json.dump(transformedTestData, f)
    with open('data/trees/test.labels.data', 'w') as f:
        json.dump(testLabelsDictionary, f)

    for i in range(0,5):
        validateLabelsDictionary, validateFeaturesDictionary = readValidateTrainFiles(i)
        validateTestLabelsDictionary, validateTestFeaturesDictionary = readValidateTestFile(i)
        validateAccuracy, transformedValidateData = evaluate_batch(validateLabelsDictionary, validateFeaturesDictionary, trees)
        validateTestAccuracy, transformedValidateTestData = evaluate_batch(validateTestLabelsDictionary, validateTestFeaturesDictionary, trees)
        with open('data/trees/split0%d.features.data' % i, 'w') as f:
            json.dump(transformedValidateData, f)
        with open('data/trees/split0%d.labels.data' % i, 'w') as f:
            json.dump(validateLabelsDictionary, f)
        with open('data/trees/split0%d.test.features.data' % i, 'w') as f:
            json.dump(transformedValidateTestData, f)
        with open('data/trees/split0%d.test.labels.data' % i, 'w') as f:
            json.dump(validateTestLabelsDictionary, f)

# The following function reads all but one cross-validation data files
def readValidateTrainFiles(heldOutK):
    global dataSetPath
    global validateSetPattern
    trainLines = []
    for i in range(0,5):
        if(i != heldOutK):
            with open((dataSetPath + validateSetPattern % (i))) as trainFile:
                trainLines = trainLines + trainFile.readlines()
    return parseLines(trainLines)

# The following function reads the validation file for cross-validations
def readValidateTestFile(fileNumber):
    global dataSetPath
    global validateSetPattern
    validateData = {}
    with open((dataSetPath + validateSetPattern % (fileNumber))) as validateFile:
        validateLines = validateFile.readlines()
    return parseLines(validateLines)

# The following three functions read the training, development, and test data sets
def readTrainFile():
    global dataSetPath
    global trainSetName
    with open(dataSetPath + trainSetName) as trainFile:
        trainLines = trainFile.readlines()
    return parseLines(trainLines)

def readTestFile():
    global dataSetPath
    global testSetName
    with open(dataSetPath + testSetName) as trainFile:
        trainLines = trainFile.readlines()
    return parseLines(trainLines)

# The following functions takes lines from a data file and parses them into labels, feature attributes and their values
# It returns two dictionaries with common keys, one contains the feature vector, and one contains the label
def parseLines(trainLines):
    labelsDictionary = {}
    featuresDictionary = {}
    trainDataList = [line.strip() for line in trainLines]
    for dataIndex in range(0,len(trainDataList)):
        data = trainDataList[dataIndex]
        dataParts = data.split(' ')
        label = -1
        if(dataParts[0] == '1'):
            label = 1
        if(len(dataParts) > 1):
            featureVector = {int(attribute.strip().split(':')[0]): int(attribute.strip().split(':')[1]) for attribute in dataParts[1:]}
        else:
            featureVector = {}
        labelsDictionary[dataIndex] = label
        featuresDictionary[dataIndex] = featureVector
    return labelsDictionary, featuresDictionary

# This function evaluates a batch of trees against a data set by majority voting among the trees
def evaluate_batch(labelsDataSet, featuresDataSet, trees):
    totalSamples = len(labelsDataSet)
    correctSamples = 0
    transformedFeaturesSet = {}
    for example in labelsDataSet:
        transformedFeature = {}
        vote_tally = 0
        index = 1
        for tree in trees:
            currentTree = tree
            while(not currentTree.prediction):
                currentFeature = currentTree.feature
                sampleFeatureValue = 0
                if currentFeature in featuresDataSet[example]:
                    sampleFeatureValue = 1
                for childTree in currentTree.children:
                    if(childTree.featureValue == sampleFeatureValue):
                        currentTree = childTree
                        break
            vote_tally += currentTree.prediction
            transformedFeature[index] = currentTree.prediction
            index += 1
        transformedFeature[0] = 1
        transformedFeaturesSet[example] = transformedFeature
        predictionValue = 1
        if vote_tally < 0:
            predictionValue = -1
        if labelsDataSet[example] == predictionValue:
            correctSamples += 1
    return (correctSamples/totalSamples)*100, transformedFeaturesSet

# This method returns a decision tree based upon the data set with given depth
def id3Algorithm(dataSet, remainingFeatures, featureValue, level, maxDepth):
    global trainFeaturesDictionary

    uniformLabels, label = allUniformLabels(dataSet)
    if(uniformLabels):
        newTree = DecisionTree(label, None, featureValue, level)
        return newTree
    elif(len(remainingFeatures) == 0):
        majorityLabel = findMajorityLabel(dataSet)
        newTree = DecisionTree(majorityLabel, None, featureValue, level)
        return newTree
    elif(level >= maxDepth):
        majorityLabel = findMajorityLabel(dataSet)
        newTree = DecisionTree(majorityLabel, None, featureValue, level)
        return newTree
    else:
        bestFeature = findBestRemainingFeature(dataSet, remainingFeatures)
        leftoverFeatures = [x for x in remainingFeatures]
        leftoverFeatures.remove(bestFeature)
        
        newTree = DecisionTree(None, bestFeature, featureValue, level)

        positiveDataSet = []
        negativeDataSet = []
        for name in dataSet:
            if(bestFeature in trainFeaturesDictionary[name]):
                positiveDataSet.append(name)
            else:
                negativeDataSet.append(name)
         
        if(len(positiveDataSet) == 0):
            commonLabel = findMajorityLabel(dataSet)
            positiveTree = DecisionTree(commonLabel, None, 1, level+1)
        else:
            positiveTree = id3Algorithm(positiveDataSet, leftoverFeatures, 1, level+1, maxDepth)
        
        if(len(negativeDataSet) == 0):
            commonLabel = findMajorityLabel(dataSet)
            negativeTree = DecisionTree(commonLabel, None, 0, level+1)
        else:
            negativeTree = id3Algorithm(negativeDataSet, leftoverFeatures, 0, level+1, maxDepth)
        
        positiveTree.parent = newTree
        negativeTree.parent = newTree
        newTree.children = [positiveTree, negativeTree]
        return newTree

# This method checks for data sets with uniform labeling
def allUniformLabels(dataSet):
    global trainLabelsDictionary

    label = trainLabelsDictionary[dataSet[0]]
    for name in dataSet:
        if(trainLabelsDictionary[name] != label):
            return False, label
    return True, label

# This data set finds the majority label of a data set
def findMajorityLabel(dataSet):
    numPositiveSamples = len(getPositiveSamples(dataSet))
    numNegativeSamples = len(getNegativeSamples(dataSet))
    if(numPositiveSamples >= numNegativeSamples):
        return 1
    else:
        return -1

# This method finds the best remaining feature based upon information gain
def findBestRemainingFeature(dataSet, features):
    if(len(features) == 1):
        return features[0]
    bestFeature = -1
    bestFeatureIG = -1000.0
    numTotalPositiveLabelSamples = len(getPositiveSamples(dataSet))
    numTotalNegativeLabelSamples = len(getNegativeSamples(dataSet))
    totalEntropy = entropy(numTotalPositiveLabelSamples, numTotalNegativeLabelSamples)
    for feature in features:
        featureIG = informationGain(dataSet,feature,totalEntropy)
        if(featureIG > bestFeatureIG):
            bestFeature = feature
            bestFeatureIG = featureIG
    return bestFeature

# This method calculates information gain of a feature over a data set
def informationGain(dataSet, feature, totalEntropy):
    global trainFeaturesDictionary

    positiveFeatureDataSet = []
    negativeFeatureDataSet = []
    for example in dataSet:
        if feature in trainFeaturesDictionary[example]:
            positiveFeatureDataSet.append(example)
        else:
            negativeFeatureDataSet.append(example)
    
    numTotalSamples = len(dataSet)
    numPositiveFeatureSamples = len(positiveFeatureDataSet)
    numNegativeFeatureSamples = len(negativeFeatureDataSet)
    
    positiveFeatureEntropy = 0
    if(numPositiveFeatureSamples > 0):
        numPositiveFeaturePositiveLabelSamples = len(getPositiveSamples(positiveFeatureDataSet))
        numPositiveFeatureNegativeLabelSamples = len(getNegativeSamples(positiveFeatureDataSet))
        positiveFeatureEntropy = entropy(numPositiveFeaturePositiveLabelSamples, numPositiveFeatureNegativeLabelSamples)

    negativeFeatureEntropy = 0
    if(numNegativeFeatureSamples > 0):
        numNegativeFeaturePositiveLabelSamples = len(getPositiveSamples(negativeFeatureDataSet))
        numNegativeFeatureNegativeLabelSamples = len(getNegativeSamples(negativeFeatureDataSet))
        negativeFeatureEntropy = entropy(numNegativeFeaturePositiveLabelSamples, numNegativeFeatureNegativeLabelSamples)

    gain = totalEntropy - ((numPositiveFeatureSamples/numTotalSamples)*positiveFeatureEntropy 
                            + (numNegativeFeatureSamples/numTotalSamples)*negativeFeatureEntropy)
    
    return gain

# This method calculates entropy for a data set
def entropy(numPositives, numNegatives):
    total = numPositives + numNegatives
    positiveFraction = numPositives/total
    negativeFraction = numNegatives/total
    if(positiveFraction == negativeFraction):
        return 1
    if(positiveFraction == 0 or negativeFraction == 0):
        return 0
    else:    
        positiveLog = positiveFraction*math.log(positiveFraction, 2)
        negativeLog = negativeFraction*math.log(negativeFraction, 2)
        totalEntropy = -positiveLog - negativeLog
        return totalEntropy

# This method returns all positive samples of a data set
def getPositiveSamples(dataSet):
    global trainLabelsDictionary

    positiveSamples = []
    for example in dataSet:
        if(trainLabelsDictionary[example] == 1):
            positiveSamples.append(example)
    return positiveSamples

# This method returns all negative samples of a data set
def getNegativeSamples(dataSet):
    global trainLabelsDictionary

    negativeSamples = []
    for example in dataSet:
        if(trainLabelsDictionary[example] == -1):
            negativeSamples.append(example)
    return negativeSamples

# This method prints a decision tree object
def printDecisionTree(tree):
    print(tree)
    for child in tree.children:
        printDecisionTree(child)

# This class represents a decision tree
class DecisionTree:
    prediction = None
    feature = None
    featureValue = None
    children = []
    parent = None
    level = None

    def __init__(self, prediction, feature, featureValue, level):
        self.prediction = prediction
        self.feature = feature
        self.featureValue = featureValue
        self.level = level

    def __str__(self):
        printString = ''
        if(self.prediction):
            printString += ('Prediction: %d\n' % (self.prediction))
        if(self.feature):
            printString += ('Feature: %d\n' % (self.feature))
        if(self.featureValue):
            printString += ('ParentFeatureValue: %d\n' % (self.featureValue))
        if(self.level):
            printString += ('Level: %d\n' % (self.level))
        printString += '\n'
        return printString



if __name__ == '__main__':
    main(sys.argv[1])