

class ClassificationMethod:
  def __init__(self, labels):
    self.labels = labels
    
    
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    abstract
    
  def classify(self, data):
    abstract