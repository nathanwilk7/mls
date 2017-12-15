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

# The following is a list of hyper-parameters that will be used to represent smoothing
smoothingHyperParams=[2, 1.5, 1, 0.5]

# This function is the orchestrator to call and collect results from naive bayes.
def main(path):
    global trainLabelsDictionary
    global testLabelsDictionary
    global trainFeaturesDictionary
    global testFeaturesDictionary
    global smoothingHyperParams
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
    numFeatures = max(features)

    print('\n---------------THESE ARE THE STATISTICS FOR THE DATA SETS---------------')

    print('The size of the training set is %d' % (len(trainLabelsDictionary)))
    print('The size of the test set is %d' % (len(testLabelsDictionary)))

    testMajorityLabel, testMajorityPercentage = getMajorityInfo(testLabelsDictionary)

    print('The majority label for test is %d, with majority percentage %f' % (testMajorityLabel, testMajorityPercentage))

    print('\n---------------THESE ARE THE RESULTS FOR THE NAIVE BAYES---------------')
    
    # We need to obtain the same info about the best smoothing, this time for naive bayes
    hyperParameterResults = {}
    for smoothingParam in smoothingHyperParams:
        validateAccuracyResults = {}
        for i in range(0,5):
            validateTrainLabelsDictionary, validateTrainFeaturesDictionary = readValidateTrainFiles(i)
            bayesLearner = train_naiveBayes(validateTrainLabelsDictionary, validateTrainFeaturesDictionary, smoothingParam)

            validateLabelsDictionary, validateFeaturesDictionary = readValidateTestFile(i)
            accuracy = evaluate_naiveBayes(bayesLearner, validateLabelsDictionary, validateFeaturesDictionary)
            validateAccuracyResults[i] = accuracy
        averageAccuracy = getAverage(validateAccuracyResults)
        standardDeviation = statistics.pstdev(list(validateAccuracyResults.values()))
        print('The average accuracy for hyper-parameter (Smoothing) = %f is %f' % (smoothingParam, averageAccuracy))
        print('The standard deviation of cross validation results for hyper-parameter (Smoothing) = %f is %f' % (smoothingParam, standardDeviation))
        hyperParameterResults[smoothingParam] = averageAccuracy
    bestSmoothingParam = findParamWithMaxAccuracy(hyperParameterResults)
    print('The best hyper-parameter (Smoothing) = %f \n' % (bestSmoothingParam))

    # Once again we will use our best hyper-parameter info to find the best number of epochs and evaluate against test data
    bayesLearner = train_naiveBayes(trainLabelsDictionary, trainFeaturesDictionary, bestSmoothingParam)
    trainAccuracy = evaluate_naiveBayes(bayesLearner, trainLabelsDictionary, trainFeaturesDictionary)
    print('Train Accuracy is %f' % trainAccuracy)
    testAccuracy = evaluate_naiveBayes(bayesLearner, testLabelsDictionary, testFeaturesDictionary)
    print('Test Accuracy is %f' % testAccuracy)

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
            featureVector = {int(attribute.strip().split(':')[0]): int(attribute.strip().split(':')[1]) for attribute in dataParts[1:]}
        else:
            featureVector = {}
        labelsDictionary[dataIndex] = label
        featuresDictionary[dataIndex] = featureVector
    return labelsDictionary, featuresDictionary

# This function trains a naive bayes learner and returns an object representing the learner
def train_naiveBayes(labelsDataSet, featuresDataSet, smoothingParam):
    global numFeatures
    positiveCount, positivePrior, negativeCount, negativePrior = getPriorsAndCounts(labelsDataSet)
    positiveLabelPositiveFeatureProbabilities = {}
    negativeLabelPositiveFeatureProbabilities = {}
    positiveLabelNegativeFeatureProbabilities = {}
    negativeLabelNegativeFeatureProbabilities = {}
    for feature in range(1,numFeatures+1):
        positiveLabelPositiveFeature = 0
        negativeLabelPositiveFeature = 0
        positiveLabelNegativeFeature = 0
        negativeLabelNegativeFeature = 0
        for example in featuresDataSet:
            if feature in featuresDataSet[example]:
                if labelsDataSet[example] == 1:
                    positiveLabelPositiveFeature += 1
                else:
                    negativeLabelPositiveFeature += 1
            else:
                if labelsDataSet[example] == 1:
                    positiveLabelNegativeFeature += 1
                else:
                    negativeLabelNegativeFeature += 1
        positiveLabelPositiveLikelihood = (positiveLabelPositiveFeature + smoothingParam) / (positiveCount + 2*smoothingParam)
        negativeLabelPositiveLikelihood = (negativeLabelPositiveFeature + smoothingParam) / (negativeCount + 2*smoothingParam)
        positiveLabelNegativeLikelihood = (positiveLabelNegativeFeature + smoothingParam) / (positiveCount + 2*smoothingParam)
        negativeLabelNegativeLikelihood = (negativeLabelNegativeFeature + smoothingParam) / (negativeCount + 2*smoothingParam)
        positiveLabelPositiveFeatureProbabilities[feature] = positiveLabelPositiveLikelihood 
        negativeLabelPositiveFeatureProbabilities[feature] = negativeLabelPositiveLikelihood
        positiveLabelNegativeFeatureProbabilities[feature] = positiveLabelNegativeLikelihood
        negativeLabelNegativeFeatureProbabilities[feature] = negativeLabelNegativeLikelihood
    bayesLearner = NaiveBayes(positivePrior, negativePrior, positiveLabelPositiveFeatureProbabilities, negativeLabelPositiveFeatureProbabilities, positiveLabelNegativeFeatureProbabilities, negativeLabelNegativeFeatureProbabilities)
    return bayesLearner

# This method evaluates a naive bayes learner object and returns the accuracy
def evaluate_naiveBayes(bayesLearner, labelsDataSet, featuresDataSet):
    numSamples = len(labelsDataSet)
    numCorrect = 0
    for example in featuresDataSet:
        positivePrediction = math.log(bayesLearner.positivePrior)
        negativePrediction = math.log(bayesLearner.negativePrior)
        for feature in range(1, numFeatures+1):
            if feature in featuresDataSet[example]:
                positivePrediction += math.log(bayesLearner.positiveLabelPositiveProbabilities[feature])
                negativePrediction += math.log(bayesLearner.negativeLabelPositiveProbabilities[feature])
            else:
                positivePrediction += math.log(bayesLearner.positiveLabelNegativeProbabilities[feature])
                negativePrediction += math.log(bayesLearner.negativeLabelNegativeProbabilities[feature])
        if positivePrediction >= negativePrediction:
            if labelsDataSet[example] == 1:
                numCorrect += 1
        else:
            if labelsDataSet[example] == -1:
                numCorrect += 1
    return (numCorrect/numSamples)*100

# This function returns the priors and counts of the overall data set
def getPriorsAndCounts(labelsDataSet):
    numSamples = len(labelsDataSet)
    numPositiveSamples = 0
    numNegativeSamples = 0
    for sample in labelsDataSet:
        if labelsDataSet[sample] == 1:
            numPositiveSamples += 1
        else:
            numNegativeSamples += 1
    return numPositiveSamples, (numPositiveSamples/numSamples), numNegativeSamples, (numNegativeSamples/numSamples)

# This class represents a naive bayes learner
class NaiveBayes:
    positivePrior = 1.0
    negativePrior = 0.0
    positiveLabelPositiveProbabilities = None
    negativeLabelPositiveProbabilities = None
    positiveLabelNegativeProbabilities = None
    negativeLabelNegativeProbabilities = None 

    def __init__(self, positivePrior, negativePrior, positiveLabelPositiveProbabilities, negativeLabelPositiveProbabilities, positiveLabelNegativeProbabilities, negativeLabelNegativeProbabilities):
        self.positivePrior = positivePrior
        self.negativePrior = negativePrior
        self.positiveLabelPositiveProbabilities = positiveLabelPositiveProbabilities
        self.negativeLabelPositiveProbabilities = negativeLabelPositiveProbabilities
        self.positiveLabelNegativeProbabilities = positiveLabelNegativeProbabilities
        self.negativeLabelNegativeProbabilities = negativeLabelNegativeProbabilities
        

if __name__ == '__main__':
    main(sys.argv[1])