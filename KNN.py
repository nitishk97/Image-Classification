import util
import numpy
import math
import statistics
import tracemalloc
import time
PRINT = True


class KNNClassifier:
  def __init__( self, labels, k=10):
    self.labels = labels
    self.type = "kNN"
    self.k = k
    self.weights = {}
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    self.trainingData = self.downscaleDataFunction(trainingData)
    self.trainingLabels = trainingLabels
    


  def downscaleDataFunction(self, datum_list):
    DATA_HEIGHT, DATA_WIDTH = 0,0
    BLOCK_HEIGHT, BLOCK_WIDTH = 0,0
    BLOCK_ROWS, BLOCK_COLS = 0,0
    if 2 in self.labels:
      DATA_HEIGHT, DATA_WIDTH = 28,28
      BLOCK_HEIGHT, BLOCK_WIDTH = 4,4
      BLOCK_ROWS, BLOCK_COLS = 7,7

    else:
      DATA_HEIGHT, DATA_WIDTH = 70,60
      BLOCK_HEIGHT, BLOCK_WIDTH = 7,6
      BLOCK_ROWS, BLOCK_COLS = 10,10

    downscaledDataAll = []
    for data in datum_list:
      downscaledData = util.Counter()
      for i_big in range(BLOCK_ROWS):
        for j_big in range(BLOCK_COLS):
          isFeature = 0

          for i_small in range(BLOCK_HEIGHT):
            if isFeature:
              break
            for j_small in range(BLOCK_WIDTH):
              if data[( i_big*BLOCK_HEIGHT + i_small , j_big*BLOCK_WIDTH + j_small )] == 1:
                isFeature = 1
                break

          downscaledData[(i_big,j_big)] = isFeature

      downscaledDataAll.append(downscaledData)

    return downscaledDataAll

  def findDistance(self, test_datum, train_data):
    if True:
      x = test_datum - train_data
      return numpy.sum(numpy.abs([x[value] for value in x]))
    
  def classify(self, data ):

    data = self.downscaleDataFunction(data)

    guesses = []
    for datum in data:
      distanceValues = []
      for i in range(len(self.trainingData)):
        distanceValues.append(  (self.findDistance(datum,self.trainingData[i]), i)  ) 

      distanceValues.sort()
      distanceValues = distanceValues[:self.k]

      bestK_labels = []
      for distance in distanceValues:
        bestK_labels.append(self.trainingLabels[distance[1]])

      try:
        guesses.append(statistics.mode(bestK_labels))
      except:
        guesses.append(bestK_labels[0])


    return guesses


