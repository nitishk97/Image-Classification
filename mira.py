import util
PRINT = True

class MiraClassifier:

  def __init__( self, labels, max_iterations):
    self.labels = labels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.labels = labels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    self.weights = {}
    for label in self.labels:
      self.weights[label] = util.Counter() 
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):      
    self.features = list(trainingData[0].keys()) 
    
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    util.raiseNotDefined()

  def classify(self, data ):
  
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.labels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighOddsFeatures(self, label1, label2):
    featuresOdds = []
    return featuresOdds
