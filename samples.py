import util
import random

DATUM_WIDTH = 0
DATUM_HEIGHT = 0

class Datum:
  def __init__(self, data,width,height):
    DATUM_HEIGHT = height
    DATUM_WIDTH=width
    self.height = DATUM_HEIGHT
    self.width = DATUM_WIDTH
    if data == None:
      data = [[' ' for i in range(DATUM_WIDTH)] for j in range(DATUM_HEIGHT)] 
    self.pixels = util.arrayInvert(convertToInteger(data)) 
    
  def getPixel(self, column, row):
    return self.pixels[column][row]
      
  def getPixels(self):
    return self.pixels    
      
  def getAsciiString(self):
    rows = []
    data = util.arrayInvert(self.pixels)
    for row in data:
      ascii = list(map(asciiGrayscaleConversionFunction, row))
      rows.append( "".join(ascii) )
    return "\n".join(rows)
    
  def __str__(self):
    return self.getAsciiString()
    


# Data processing, cleanup and display functions
    
def loadDataFile(filename, n,width,height,isRandom=False):
  DATUM_WIDTH=width
  DATUM_HEIGHT=height
  fin = readlines(filename)
  fin.reverse()
  items = []

  dataAmountInFile = len(fin) // DATUM_HEIGHT
  chosenList = []
  if isRandom:
    chosenList = random.sample(range(0, dataAmountInFile), n)
  else:
    chosenList = range(n)

  for i in chosenList:
    data = []
    startValue = -(i*height + 1) 
    for j in range(height):
      data.append(list(fin[startValue-j]))
    if len(data[0]) < DATUM_WIDTH-1:
      print("Truncating at %d examples (maximum)" % i)
      break
    items.append(Datum(data,DATUM_WIDTH,DATUM_HEIGHT))
  return (items, chosenList)


import zipfile
import os
def readlines(filename):
  if(os.path.exists(filename)): 
    return [l[:-1] for l in open(filename).readlines()]
  else: 
    z = zipfile.ZipFile('data.zip')
    return z.read(filename).split('\n')
    
def loadLabelsFile(filename, chosenList):
  fin = readlines(filename)
  labels = []

  for value in chosenList:
    labels.append(int(fin[value]))

  return labels
  
def asciiGrayscaleConversionFunction(value):
  if(value == 0):
    return ' '
  elif(value == 1):
    return '+'
  elif(value == 2):
    return '#'    
    
def IntegerConversionFunction(character):
  if(character == ' '):
    return 0
  elif(character == '+'):
    return 1
  elif(character == '#'):
    return 2    

def convertToInteger(data):
  if type(data) != type([]):
    return IntegerConversionFunction(data)
  else:
    return list(map(convertToInteger, data))