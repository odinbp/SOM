import math
import random
import csv
import matplotlib.pyplot as plt
import mnist_basics as mb

testVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253,
    253, 225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107,
      253, 253, 205, 11, 0, 43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24,
             114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226,
               253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
testClass = 5


class SOMMnist(object):

	def __init__(self, n_iterations=100, lr0 = None, tlr = None, size0 = None, tsize = None, multiplier = 2):
		#something
		if lr0 == None:
			self.lr0 = 0.2
		else:
			self.lr0 = float(lr0)
		if size0 == None:
			self.size0 = max(m,n)/2.0
		else:
			self.size0 = float(size0*multiplier)
		self.lr = self.lr0
		self.size = self.size0
		self.tlr = tlr
		self.tsize = tsize
		self.multiplier = multiplier

		self.n_iterations = n_iterations

		self.sizeChanges = [size0]
		self.lrateChanges = [lr0]

	def init_neurons(self, d, count):
		weights = [[random.uniform(0.0,1.0) for i in range(d)] for j in range(100)] 
		#grid = [[0]*10]*10
		predicts = []
		for i in range(100):
			predicts.append([])
		return [weights, predicts]


	def discriminant(self, iv, weight):
		d = 0
		for i in range(len(iv)):
			d +=  (iv[0][i]-weight[i])**2
		return math.sqrt(d)

	def neighbourhood(self, distance, size):
		p = 2*size**2
		if p == 0:
			return 0
		return math.exp((-distance**2)/p)
		

	def som(self, neurons, inputs, iterations, k):
		
		#You need only compute total distance (D) and show diagrams at every k steps
		#print(neurons, "som") #feil
		for i in range(iterations+1):
			self.som_one_step(neurons, inputs, i)
			
	def som_one_step(self, neurons, inputs, iter):

		#Pick a random input vector
		iv = inputs[random.randint(0,len(inputs)-1)]
		#Find winner of cometition
		winningIndex, winner = self.find_winner(iv, neurons[0])
		neurons[1][winningIndex].append(iv[1])
		iv = iv[0]
		for neuronIndex, neuron in enumerate(neurons[0]):
			d = self.grid_distance(neuronIndex, winningIndex)
			tn = self.neighbourhood(d, self.size)
			for i in range(len(neuron)):
				neuron[i] += self.lr*tn*(iv[i]-neuron[i])

		self.size_decay(iter)
		self.learning_decay(iter)

	def find_winner(self, iv, neurons):
		mi,mn = -100,[255]*786
		for index,neuron in enumerate(neurons):
			#print(neuron, "BOOOOOYEAH") #denne blir feil
			if self.discriminant(iv,neuron) < self.discriminant(iv,mn):
				mi = index
				mn = neuron
		return mi,mn

	#Manhattan distance, but far from perfect. At the moment it works because the grid is 10x10
	def grid_distance(self, neuronIndex, winningIndex):
		neuronIndex = str(neuronIndex)
		winningIndex = str(winningIndex)

		if len(neuronIndex) == 1:
			neuronIndex = '0'+ neuronIndex
		if len(winningIndex) == 1:
			winningIndex = '0'+ winningIndex

		NIX = int(neuronIndex[0])
		NIY = int(neuronIndex[1])
		WIX = int(winningIndex[0])
		WIY = int(winningIndex[1])

		return (abs(NIX-WIX) + abs(NIY - WIY))


	def size_decay(self, t):
		self.sizeChanges.append(self.size0*math.exp(-t/self.tsize)-self.size)
		self.size = self.size0*math.exp(-t/self.tsize)
		

	def learning_decay(self, t):
		self.lrateChanges.append(self.lr0*math.exp(-t/self.tlr)-self.lr)
		self.lr = self.lr0*math.exp(-t/self.tlr)

	def read_data(self, noOfImages):
		images = []
		a,b = mb.load_all_flat_cases()
		for i in range(noOfImages):
			images.append([a[i],b[i]])

		return noOfImages, images

	#must be generalized. Not sure if 255 is max for all images
	def normalize(self, data):
		scale = 255
		for d in range(len(data)):
			for e in range(len(data[d][0])):
				data[d][0][e] = data[d][0][e]/scale 
		return scale,data

	def main(self):
		noOfImages, images = self.read_data(50000)
		print("Done with loading")
		scaleFactor, scaled = self.normalize(images)
		#print(len(images[0])) == 2
		#print(images[5], "images")
		neurons = self.init_neurons(len(images[0][0]), noOfImages)
		#print(neurons, "main")
		self.som(neurons = neurons, inputs = scaled, iterations = self.n_iterations, k = 50)

		average = []
		for i in range(len(neurons[1])):
			lengthOfList = len(neurons[1][i])
			if lengthOfList == 0:
				average.append(None)
			else:
				total = 0
				for i in neurons[1][i]:
					total += i

				average.append(round(total/lengthOfList))

		
		a, b = self.find_winner([testVector, testClass], neurons[0])
		print(average[a])

		#return neurons[1]
		return average

som = SOMMnist(n_iterations=1000,lr0 = 1, tlr = 1000, size0 = 70/10, tsize = 200, multiplier = 10) 
grid = som.main()

index = 0
for i in range(10):
	lal = []
	for j in range(10):
		lal.append(grid[index])
		index += 1
	print(lal)
