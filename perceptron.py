# Implementing Perceptron
import util
PRINT = True

class PerceptronClassifier:
  def __init__( self, labels, max_iterations):
    self.labels = labels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in labels:
      self.weights[label] = util.Counter()

  def setWeights(self, weights):
    assert len(weights) == len(self.labels);
    self.weights == weights;
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    self.features = list(trainingData[0].keys())
    for iteration in range(self.max_iterations):
      print("Starting iteration %d..." % iteration)
      for i in range(len(trainingData)):
        vectors = util.Counter()
        for label in self.labels:
          vectors[label] = self.weights[label] * trainingData[i]
        best_guess_label = vectors.argMax()
        if trainingLabels[i] != best_guess_label:
          self.weights[trainingLabels[i]] += trainingData[i]
          self.weights[best_guess_label] -= trainingData[i]

    
  def classify(self, data ):
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.labels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighWeightFeatures(self, label):
    featuresWeights = []
    util.raiseNotDefined()
    return featuresWeights
