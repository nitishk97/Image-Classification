import naiveBayes
import perceptron
import KNN
import samples
import sys
import util
import time

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def enhancedFeatureExtractorDigit(datum):
  features =  basicFeatureExtractorDigit(datum)
  return features


def contestFeatureExtractorDigit(datum):
  features =  basicFeatureExtractorDigit(datum)
  return features

def enhancedFeatureExtractorFace(datum):
  features =  basicFeatureExtractorFace(datum)
  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          break

class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print("new features:", pix)
            continue
      print(image)  

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)
  
  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=[ 'nb', 'naiveBayes', 'perceptron', 'kNN', 'kNearestNeighbors', 'minicontest'], default='naiveBayes')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
  parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
  parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
  parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
  parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
  parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
  parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=1.0)
  parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=2, type="int")
  parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}
  
  print("Doing classification")
  print("--------------------")
  print("classifier:\t\t" + options.classifier)
  print("data:\t\t" + options.data)
  if not options.classifier == 'minicontest':
    print("using enhanced features?:\t" + str(options.features))
  else:
    print("using minicontest feature extractor")
  print("training set size:\t" + str(options.training))
  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorDigit
    else:
      featureFunction = basicFeatureExtractorDigit
    if (options.classifier == 'minicontest'):
      featureFunction = contestFeatureExtractorDigit
  elif(options.data=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorFace
    else:
      featureFunction = basicFeatureExtractorFace      
  else:
    print("Unknown dataset", options.data)
    print(USAGE_STRING)
    sys.exit(2)
    
  if(options.data=="digits"):
    labels = list(range(10))
  else:
    labels = list(range(2))
    
  if options.training <= 0:
    print("Training set size should be a positive integer (you provided: %d)" % options.training)
    print(USAGE_STRING)
    sys.exit(2)
    
  if options.smoothing <= 0:
    print("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
    print(USAGE_STRING)
    sys.exit(2)
    
  if options.odds:
    if options.label1 not in labels or options.label2 not in labels:
      print("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
      print(USAGE_STRING)
      sys.exit(2)

  if(options.classifier == "kNN" or options.classifier == "kNearestNeighbors"):
    classifier = KNN.KNNClassifier(labels)
  elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
    classifier = naiveBayes.NaiveBayesClassifier(labels)
    classifier.setSmoothing(options.smoothing)
    if (options.autotune):
        print("using automatic tuning for naivebayes")
        classifier.automaticTuning = True
    else:
        print("using smoothing parameter k=%f for naivebayes" %  options.smoothing)
  elif(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(labels,options.iterations)
  elif(options.classifier == 'minicontest'):
    import minicontest
    classifier = minicontest.contestClassifier(labels)
  else:
    print("Unknown classifier:", options.classifier)
    print(USAGE_STRING)
    
    sys.exit(2)

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage
  
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default naiveBayes classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

def runClassifier(args, options):
  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']
  numTraining = options.training
  numTest = options.test

  if(options.data=="faces"):
    rawTrainingData, chosenList = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT,True)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", chosenList)
    rawTestData, chosenList = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", chosenList)
  else:
    rawTrainingData, chosenList = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT,True)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", chosenList)
    rawTestData, chosenList = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", chosenList)
    
  
  print("Extracting features...")
  trainingData = list(map(featureFunction, rawTrainingData))
  testData = list(map(featureFunction, rawTestData))
  
  print("Training...")
  validationData, validationLabels = [0], [0]
  start_time = time.time()
  classifier.train(trainingData, trainingLabels, validationData, validationLabels)
  train_time = time.time() - start_time
  print(train_time, end='')
  print (" (training time)")
  print("Testing...")
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print(str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels)))
  print("")
  analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
  
  if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
    label1, label2 = options.label1, options.label2
    features_odds = classifier.findHighOddsFeatures(label1,label2)
    if(options.classifier == "naiveBayes" or options.classifier == "nb"):
      string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
    else:
      string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)    
      
    print(string3)
    printImage(features_odds)

  if((options.weights) & (options.classifier == "perceptron")):
    for l in classifier.labels:
      features_weights = classifier.findHighWeightFeatures(l)
      print(("=== Features with high weight for label %d ==="%l))
      printImage(features_weights)

if __name__ == '__main__':
  # Read input
  args, options = readCommand( sys.argv[1:] ) 
  # Run classifier
  runClassifier(args, options)