import json
import itertools
import numpy as np
import numpy.matlib
import numpy.linalg
import math
import random
import statistics
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os

# Set the number of features available
numFeatures = 0

# The following are global dictionaries used to store the data from various sets (test, train, dev, cross-validate)
trainLabelsDictionary = {}
trainFeaturesDictionary = {}
validateLabelsDictionary = {}
validateFeaturesDictionary = {}
testLabelsDictionary = {}
testFeaturesDictionary = {}

# The following is a list of hyper-parameters that will be used to represent both learning rate and tradeoff
learningRateHyperParams=[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
tradeoffHyperParams=[10000, 1000, 100, 10, 1, 0.1]

# This function is the orchestrator to call and collect results from perceptron variants.
def main():
    global trainLabelsDictionary
    global testLabelsDictionary
    global trainFeaturesDictionary
    global testFeaturesDictionary
    global learningRateHyperParams
    global tradeoffHyperParams 
    global validateLabelsDictionary
    global validateFeaturesDictionary
    global numFeatures

    try:
        os.stat('data/trees')
    except:
        os.mkdir('data/trees')

    with open('data/trees/train.features.data', 'r') as f:
        trainFeaturesDictionary = json.load(f, object_hook=jsonIntFeatures)
    with open('data/trees/train.labels.data', 'r') as f:
        trainLabelsDictionary = json.load(f, object_hook=jsonIntLabels)
    with open('data/trees/test.features.data', 'r') as f:
        testFeaturesDictionary = json.load(f, object_hook=jsonIntFeatures)
    with open('data/trees/test.labels.data', 'r') as f:
        testLabelsDictionary = json.load(f, object_hook=jsonIntLabels)

    features = []
    for featureVector in trainFeaturesDictionary.values():
        features += list(featureVector.keys())
    for featureVector in testFeaturesDictionary.values():
        features += list(featureVector.keys())
    numFeatures = max(features)+1

    print('\n---------------THESE ARE THE STATISTICS FOR THE DATA SETS---------------')

    print('The size of the training set is %d' % (len(trainLabelsDictionary)))
    print('The size of the test set is %d' % (len(testLabelsDictionary)))

    testMajorityLabel, testMajorityPercentage = getMajorityInfo(testLabelsDictionary)

    print('The majority label for test is %d, with majority percentage %f' % (testMajorityLabel, testMajorityPercentage))

    print('\n---------------THESE ARE THE RESULTS FOR THE LOGISTIC REGRESSION OVER TREES---------------')
    
    # We need to obtain the same info about the best learning rate and tradeoff
    hyperParameterResults = {}
    for learningParam, tradeoffParam in itertools.product(learningRateHyperParams, tradeoffHyperParams):
        validateAccuracyResults = {}
        for i in range(0,5):
            with open('data/trees/split0%d.features.data' % i, 'r') as f:
                validateTrainFeaturesDictionary = json.load(f, object_hook=jsonIntFeatures)
            with open('data/trees/split0%d.labels.data' % i, 'r') as f:
                validateTrainLabelsDictionary = json.load(f, object_hook=jsonIntLabels)
            weightVector = cross_logisticRegression(validateTrainLabelsDictionary, validateTrainFeaturesDictionary, 10, learningParam, tradeoffParam)

            with open('data/trees/split0%d.test.features.data' % i, 'r') as f:
                validateFeaturesDictionary = json.load(f, object_hook=jsonIntFeatures)
            with open('data/trees/split0%d.test.labels.data' % i, 'r') as f:
                validateLabelsDictionary = json.load(f, object_hook=jsonIntLabels)
            accuracy = evaluatePerformance(weightVector, validateLabelsDictionary, validateFeaturesDictionary)
            validateAccuracyResults[i] = accuracy
        averageAccuracy = getAverage(validateAccuracyResults)
        standardDeviation = statistics.pstdev(list(validateAccuracyResults.values()))
        print('The average accuracy for hyper-parameter (Learning Rate) = %f and (Tradeoff) = %f is %f' % (learningParam, tradeoffParam, averageAccuracy))
        print('The standard deviation of cross validation results for hyper-parameter (Learning Rate) = %f and (Tradeoff) = %f is %f' % (learningParam, tradeoffParam, standardDeviation))
        hyperParameterResults[str(learningParam) + '-' + str(tradeoffParam)] = averageAccuracy
    bestParams = findParamWithMaxAccuracy(hyperParameterResults)
    bestLearningParam = float(bestParams.split('-')[0])
    bestTradeoffParam = float(bestParams.split('-')[1])
    print('The best hyper-parameter (Learning Rate) = %f and (Tradeoff) = %f\n' % (bestLearningParam, bestTradeoffParam))

    # Once again we will use our best hyper-parameter info to find the best number of epochs and evaluate against test data
    weightVector, epochAccuracy, epoch, numUpdates = logisticRegression(trainLabelsDictionary, trainFeaturesDictionary, 10, bestLearningParam, bestTradeoffParam, True)
    evaluateAlgorithm(weightVector, epochAccuracy, epoch, numUpdates)

def jsonIntLabels(data):
    if isinstance(data, dict):
        return {int(key): int(value) for key, value in data.items()}
    return data

def jsonIntFeatures(data):
    if isinstance(data, dict):
        return {int(key): jsonIntFeatures(value) for key, value in data.items()}
    return data

# This function takes info about best weight vector, given the best hyper-parameters
# It collects and prints data about the resulting evaluation of our hypothesis against data sets
def evaluateAlgorithm(weightVector, epochAccuracy, epoch, numUpdates):
    global trainLabelsDictionary
    global trainFeaturesDictionary
    global testLabelsDictionary
    global testFeaturesDictionary

    trainAccuracy = evaluatePerformance(weightVector, trainLabelsDictionary, trainFeaturesDictionary)
    print('The best epoch is %d with accuracy %f on train data' % (epoch, epochAccuracy))
    print('Total Updates On Train Set for Epoch %d is %d' % (epoch, numUpdates))
    print('Train Accuracy is %f' % trainAccuracy)
    testAccuracy = evaluatePerformance(weightVector, testLabelsDictionary, testFeaturesDictionary)
    print('Test Accuracy is %f' % testAccuracy)

# This function determines which label is a majority in a data set and gives the majority percentage
def getMajorityInfo(labelDataSet):
    numPositiveExamples = 0
    numNegativeExamples = 0
    for label in labelDataSet.values():
        if(label < 0):
            numNegativeExamples += 1
        else:
            numPositiveExamples += 1
    if(numNegativeExamples > numPositiveExamples):
        return -1, (numNegativeExamples/len(labelDataSet))*100
    return 1, (numPositiveExamples/len(labelDataSet))*100

# This function will take a sparse dictionary as input and return a full numpy vector
def getDenseVector(sparseDict):
    global numFeatures
    array = csr_matrix((list(sparseDict.values()), ([0 for x in sparseDict.keys()],list(sparseDict.keys()))),shape=(1,numFeatures)).toarray()[0]
    return array

# This function will evaluate the specific accuracy for a given weight vector and data set
def evaluatePerformance(weightVector, labelSet, featureSet):
    dataSetSize = len(featureSet)
    numCorrect = 0
    for index in range(0,dataSetSize):
        attributeVector = getDenseVector(featureSet[index])
        actualLabel = labelSet[index]
        predictedLabel = -1
        product = np.dot(weightVector,attributeVector)
        if(product >= 0):
            predictedLabel = 1
        if(actualLabel == predictedLabel):
            numCorrect += 1
    accuracy = (numCorrect/dataSetSize)*100
    return accuracy

# This function returns the average of accuracy results for cross-validation
def getAverage(accuracyResults):
    total = 0
    count = 0
    for i in list(accuracyResults.keys()):
        total += accuracyResults[i]
        count += 1
    average = total/count
    return average

# This function returns the best hyper-parameter given a set of results about accuracy for each parameter
def findParamWithMaxAccuracy(accuracyResults):
    maxAccuracy = -1
    maxAccuracyParam = -1
    for param in accuracyResults:
        accuracy = accuracyResults[param]
        if(accuracy > maxAccuracy):
            maxAccuracy = accuracy
            maxAccuracyParam = param
    return maxAccuracyParam

# The following function reads all but one cross-validation data files
def readValidateTrainFiles(heldOutK):
    trainLines = []
    for i in range(0,5):
        if(i != heldOutK):
            with open(('data/CVSplits/training0%d.data' % (i))) as trainFile:
                trainLines = trainLines + trainFile.readlines()
    return parseLines(trainLines)

# The following function reads the validation file for cross-validations
def readValidateTestFile(fileNumber):
    validateData = {}
    with open(('data/CVSplits/training0%d.data' % (fileNumber))) as validateFile:
        validateLines = validateFile.readlines()
    return parseLines(validateLines)

# The following three functions read the training, development, and test data sets
def readTrainFile():
    with open('data/speeches.train.liblinear') as trainFile:
        trainLines = trainFile.readlines()
    return parseLines(trainLines)

def readTestFile():
    with open('data/speeches.test.liblinear') as trainFile:
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
        featureVector[0] = 1
        labelsDictionary[dataIndex] = label
        featuresDictionary[dataIndex] = featureVector
    return labelsDictionary, featuresDictionary

# This method implements the logistic regression algorithm for a given number of epochs
# It returns the best feature vector among all epochs, with epoch information
# The graph input parameter specifies whether or not a graph of epoch/accuracy should be generated
def logisticRegression(labelSet, featureSet, numEpochs, learnRate, tradeoff, graph):
    global trainLabelsDictionary
    global testLabelsDictionary
    global numFeatures
    
    weightVector = np.zeros(numFeatures)
    for index in range(0,numFeatures):
        weightVector[index] = random.uniform(-0.01, 0.01)
    dataSetSize = len(featureSet)
    t = 0
    numUpdates = 0
    bestAccuracy = -1
    bestVector = []
    bestEpoch = -1
    bestEpochUpdates = -1
    accuracyDictionary = {}
    shuffleOrder = np.random.permutation(dataSetSize)
    for epoch in range(0,numEpochs):
        np.random.shuffle(shuffleOrder)
        for i in range(0,dataSetSize):
            attributeVector = getDenseVector(featureSet[shuffleOrder[i]])
            predictionValue = np.dot(weightVector,attributeVector)
            actualValue = labelSet[shuffleOrder[i]]
            dynLearnRate = learnRate/(1+t)
            if(actualValue*predictionValue < 0):
                weightVector = numpy.multiply((1-(2*dynLearnRate/tradeoff)),weightVector) + numpy.multiply((dynLearnRate*actualValue*(1/(1+math.exp(actualValue*predictionValue)))), attributeVector)
                numUpdates += 1
            t += 1
        accuracy = evaluatePerformance(weightVector, trainLabelsDictionary, trainFeaturesDictionary)
        accuracyDictionary[epoch+1] = accuracy
        if(accuracy >= bestAccuracy):
            bestAccuracy = accuracy
            bestVector = weightVector
            bestEpoch = epoch+1
            bestEpochUpdates = numUpdates
    
    if(graph):
        plt.plot(list(accuracyDictionary.keys()), list(accuracyDictionary.values()), linewidth=2, marker='o')
        plt.xlabel('Epoch ID')
        plt.ylabel('Accuracy On Train Data')
        plt.title('Learning Curve - Log over Trees')
        plt.axis([0,21, -5, 105])
        plt.xticks(list(accuracyDictionary.keys()), list(accuracyDictionary.keys()), rotation='horizontal')
        plt.savefig('log_trees.png')
        plt.clf()
    return bestVector, bestAccuracy, bestEpoch, bestEpochUpdates

# This function is a logistic regression algorithm that simply returns the weight vector
# after all epochs. It is used for the cross-validation step
def cross_logisticRegression(labelSet, featureSet, numEpochs, learnRate, tradeoff):
    global numFeatures
    weightVector = np.zeros(numFeatures)
    for index in range(0,numFeatures):
        weightVector[index] = random.uniform(-0.01, 0.01)
    dataSetSize = len(featureSet)
    t = 0
    shuffleOrder = np.random.permutation(dataSetSize)
    for epoch in range(0,numEpochs):
        np.random.shuffle(shuffleOrder)
        for i in range(0,dataSetSize):
            attributeVector = getDenseVector(featureSet[shuffleOrder[i]])
            predictionValue = np.dot(weightVector,attributeVector)
            actualValue = labelSet[shuffleOrder[i]]
            dynLearnRate = learnRate/(1+t)
            if(actualValue*predictionValue < 0):
                weightVector = numpy.multiply((1-(2*dynLearnRate/tradeoff)),weightVector) + numpy.multiply((dynLearnRate*actualValue*(1/(1+math.exp(actualValue*predictionValue)))), attributeVector)
            t += 1
    return weightVector

if __name__ == '__main__':
    main()