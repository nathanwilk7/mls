import itertools
import numpy as np
import numpy.matlib
import numpy.linalg
import math
import random
import statistics
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
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

# The following is a list of hyper-parameters that will be used to represent both learning rate and regularization
learningRateHyperParams=[10, 1, 0.1, 0.01, 0.001, 0.0001]
regularizationHyperParams=[10, 1, 0.1, 0.01, 0.001, 0.0001]

# This function is the orchestrator to call and collect results from svm.
def main(path):
    global trainLabelsDictionary
    global testLabelsDictionary
    global trainFeaturesDictionary
    global testFeaturesDictionary
    global learningRateHyperParams
    global regularizationHyperParams
    global validateLabelsDictionary
    global validateFeaturesDictionary
    global numFeatures
    global dataSetPath

    dataSetPath = path

    trainLabelsDictionary, trainFeaturesDictionary = readTrainFile()
    testLabelsDictionary, testFeaturesDictionary = readTestFile()

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

    print('\n---------------THESE ARE THE RESULTS FOR THE STOCHASTIC GRADIENT DESCENT SVM---------------')
    
    # We need to obtain the same info about the best learning rate and regularization
    hyperParameterResults = {}
    for learningParam, regularizationParam in itertools.product(learningRateHyperParams, regularizationHyperParams):
        validateAccuracyResults = {}
        for i in range(0,5):
            validateTrainLabelsDictionary, validateTrainFeaturesDictionary = readValidateTrainFiles(i)
            weightVector = cross_svm(validateTrainLabelsDictionary, validateTrainFeaturesDictionary, 10, learningParam, regularizationParam)

            validateLabelsDictionary, validateFeaturesDictionary = readValidateTestFile(i)
            accuracy = evaluatePerformance(weightVector, validateLabelsDictionary, validateFeaturesDictionary)
            validateAccuracyResults[i] = accuracy
        averageAccuracy = getAverage(validateAccuracyResults)
        standardDeviation = statistics.pstdev(list(validateAccuracyResults.values()))
        print('The average accuracy for hyper-parameter (Learning Rate) = %f and (Regularization) = %f is %f' % (learningParam, regularizationParam, averageAccuracy))
        print('The standard deviation of cross validation results for hyper-parameter (Learning Rate) = %f and (Regularization) = %f is %f' % (learningParam, regularizationParam, standardDeviation))
        hyperParameterResults[str(learningParam) + '-' + str(regularizationParam)] = averageAccuracy
    bestParams = findParamWithMaxAccuracy(hyperParameterResults)
    bestLearningParam = float(bestParams.split('-')[0])
    bestRegularizationParam = float(bestParams.split('-')[1])
    print('The best hyper-parameter (Learning Rate) = %f and (Regularization) = %f\n' % (bestLearningParam, bestRegularizationParam))

    # Once again we will use our best hyper-parameter info to find the best number of epochs and evaluate against test data
    weightVector, epochAccuracy, epoch, numUpdates = svm(trainLabelsDictionary, trainFeaturesDictionary, 10, bestLearningParam, bestRegularizationParam, True)
    evaluateAlgorithm(weightVector, epochAccuracy, epoch, numUpdates)

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
            featureVector = {int(attribute.strip().split(':')[0]): float(attribute.strip().split(':')[1]) for attribute in dataParts[1:]}
        else:
            featureVector = {}
        featureVector[0] = 1
        labelsDictionary[dataIndex] = label
        featuresDictionary[dataIndex] = featureVector
    return labelsDictionary, featuresDictionary

# This method implements the svm algorithm for a given number of epochs
# It returns the best feature vector among all epochs, with epoch information
# The graph input parameter specifies whether or not a graph of epoch/accuracy should be generated
def svm(labelSet, featureSet, numEpochs, learnRate, regularization, graph):
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
            if(actualValue*predictionValue <= 1):
                weightVector = numpy.multiply((1-dynLearnRate), weightVector) + numpy.multiply((dynLearnRate*regularization*actualValue),attributeVector)
            else:
                weightVector = numpy.multiply((1-dynLearnRate), weightVector)
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
        plt.title('Learning Curve - SVM')
        plt.axis([0,21, -5, 105])
        plt.xticks(list(accuracyDictionary.keys()), list(accuracyDictionary.keys()), rotation='horizontal')
        plt.savefig('svm.png')
        plt.clf()
    return bestVector, bestAccuracy, bestEpoch, bestEpochUpdates

# This function is a svm algorithm that simply returns the weight vector
# after all epochs. It is used for the cross-validation step
def cross_svm(labelSet, featureSet, numEpochs, learnRate, regularization):
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
            if(actualValue*predictionValue <= 1):
                weightVector = numpy.multiply((1-dynLearnRate), weightVector) + numpy.multiply((dynLearnRate*regularization*actualValue),attributeVector)
            else:
                weightVector = numpy.multiply((1-dynLearnRate), weightVector)
            t += 1
    return weightVector

if __name__ == '__main__':
    main(sys.argv[1])