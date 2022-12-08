

import util
import classificationMethod

class MostFrequentClassifier(classificationMethod.ClassificationMethod):

  def __init__(self, labels):
    self.guess = None
    self.type = "mostfrequent"
  
  def train(self, data, labels, validationData, validationLabels):
  
    counter = util.Counter()
    counter.incrementAll(labels, 1)
    self.guess = counter.argMax()
  
  def classify(self, testData):
    return [self.guess for i in testData]