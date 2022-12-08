
import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  def __init__(self, labels):
    self.labels = labels
    self.type = "naivebayes"
    self.k = 1 
    self.automaticTuning = False 
    print("Legal Labels:", self.labels)
    
  def setSmoothing(self, k):
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    self.features = list(set([ f for datum in trainingData for f in list(datum.keys()) ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

  def getFeatureCountTrue(self, feature, label):
    return self.featureCounts[label][feature]

  def getFeatureCountFalse(self, feature, label):
    return self.count_labels[label] - self.featureCounts[label][feature]
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    self.count_labels = [0 for x in self.legalLabels]

    self.featureCounts = {}
    for label in self.labels:
      self.featureCounts[label] = util.Counter() 

    counter = 0
    for i in range(len(trainingData)):
      counter += 1
      self.count_labels[trainingLabels[i]] += 1
      self.featureCounts[i] = util.Counter()
      self.featureCounts[trainingLabels[i]] += trainingData[i]

    self.dataCount = counter
    
        
  def classify(self, testData):

    guesses = []
    self.posteriors = [] 
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):

    logJoint = util.Counter()
    for label in self.labels:
      priorProb_Labels = math.log(self.count_labels[label] / self.dataCount)

      featureProb_givenLabel = 0
      for feature in datum:
        trueCount = self.getFeatureCountTrue(feature, label) + self.k
        falseCount = self.getFeatureCountFalse(feature, label) + self.k
        denominator = trueCount + falseCount

        if(datum[feature]):
          featureProb_givenLabel += math.log(trueCount / denominator)
        else:
          featureProb_givenLabel += math.log(falseCount / denominator)

      logJoint[label] = priorProb_Labels + featureProb_givenLabel
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    featuresOdds = []
    util.raiseNotDefined()

    return featuresOdds