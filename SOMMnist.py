import math
import random
import csv
import matplotlib.pyplot as plt
import mnist_basics as mb
import numpy as np
import statistics

plt.rcParams['figure.figsize'] = (11,7)

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

	def __init__(self, noOfNeuron, n_iterations=100, lr0 = None, tlr = None, size0 = None, tsize = None):
		#something
		self.lr0 = float(lr0)
		self.size0 = float(size0)
		self.lr = self.lr0
		self.size = self.size0
		self.tlr = tlr
		self.tsize = tsize

		self.n_iterations = n_iterations

		self.sizeChanges = [size0]
		self.lrateChanges = [lr0]
		self.noOfNeuron = noOfNeuron
		self.liste = self.makeList(noOfNeuron)

	def init_neurons(self, d, count):
		weights = [[random.uniform(0.0,1.0) for i in range(d)] for j in range(count*count)] 
		predicts = []
		for i in range(count*count):
			predicts.append([])
		return [weights, predicts]


	def discriminant(self, iv, weight):
		d = 0
		for i in range(len(iv)):
			d +=  (iv[i]-weight[i])**2
		return math.sqrt(d)

	def neighbourhood(self, distance, size):
		p = 2*size**2
		if p == 0:
			return 0
		return math.exp((-distance**2)/p)
		

	def som(self, neurons, inputs, iterations, k, noOfNeuron):
		
		#You need only compute total distance (D) and show diagrams at every k steps
		#print(neurons, "som") #feil
		for i in range(iterations+1):
			self.som_one_step(neurons, inputs, i, noOfNeuron)
		################
			if i%k == 0:
				average = []

				for z in range(len(neurons[1])):
					lengthOfList = len(neurons[1][z])
					if lengthOfList == 0:
						average.append(None)
					else:
						#total = 0
						total = []
						for element in neurons[1][z]:
							#total += element
							total.append(element)
						#average.append(round(total/lengthOfList))
						average.append(statistics.median(total))
				


				index = 0
				for j in range(noOfNeuron):
					output = []
					for _ in range(noOfNeuron):
						output.append(average[index])
						index += 1
					print(output)

		#######################################


	def som_one_step(self, neurons, inputs, iter, noOfNeuron):

		#Pick a random input vector
		iv = inputs[random.randint(0,len(inputs)-1)]
		#Find winner of cometition
		winningIndex, winner = self.find_winner(iv[0], neurons[0])
		neurons[1][winningIndex].append(iv[1])
		iv = iv[0]
		for neuronIndex, neuron in enumerate(neurons[0]):
			d = self.grid_distance(neuronIndex, winningIndex, noOfNeuron)
			tn = self.neighbourhood(d, self.size)
			for i in range(len(neuron)):
				neuron[i] += self.lr*tn*(iv[i]-neuron[i])

		self.size_decay(iter)
		self.learning_decay(iter)

	def find_winner(self, iv, neurons):
		mi,mn = -100,[255]*786
		for index,neuron in enumerate(neurons):
			if self.discriminant(iv,neuron) < self.discriminant(iv,mn):
				mi = index
				mn = neuron
		return mi,mn

	def findIndex(self, index,list):
		for i in range(len(list)):
			if [index] in list[i]:
				xIndex = list[i].index([index])
				yIndex = i
				break
		return (xIndex, yIndex)

	def makeList(self, noOfNeuron):
		liste = []
		for i in range(0,noOfNeuron*noOfNeuron,noOfNeuron):
			temp = []
			for j in range(noOfNeuron):
				temp.append([i+j])
			liste.append(temp)
		return liste

	def grid_distance(self, neuronIndex, winningIndex, noOfNeuron):
	
		neuronX, neuronY = self.findIndex(neuronIndex, self.liste)
		winningX, winningY = self.findIndex(winningIndex, self.liste)

		#return (abs( neuronX - winningX ) + abs( neuronY - winningY ))
		return (math.sqrt(( neuronX - winningX )**2 + abs( neuronY - winningY )**2))
		

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
		noOfImages, images = self.read_data(1000)
		scaleFactor, scaled = self.normalize(images)
		noOfNeuron = self.noOfNeuron
		neurons = self.init_neurons(len(images[0][0]), noOfNeuron)#noOfImages)
		self.som(noOfNeuron = noOfNeuron, neurons = neurons, inputs = scaled, iterations = self.n_iterations, k = 1000)
		

		average = []
		for i in range(len(neurons[1])):
			lengthOfList = len(neurons[1][i])
			if lengthOfList == 0:
				average.append(0)
			else:
				#total = 0
				total = []
				for element in neurons[1][i]:
					#total += element
					total.append(element)

				#average.append(round(total/lengthOfList))
				average.append(statistics.median(total))
		

		#Tester på testVector
		#a, b = self.find_winner(testVector, neurons[0])
		#print(average[a])

		noTrainImages, trainImages = self.read_data(100)
		correct = 0
		for i in range(100):
			a,b = self.find_winner(trainImages[i][0], neurons[0])
			if average[a] == trainImages[i][1]:
				correct += 1
		print((correct/100)*100)
		#print("prediction:", average[a], "target", trainImages[i][1])


		return average

#tlr -> learning decay, size0 -> neighbourhood, tsize - > size decay, multiplier - > size ganges med multiplier
som = SOMMnist(n_iterations=3000,lr0 = 0.1, tlr = 2000, size0 = 40, tsize = 500, noOfNeuron = 10) 
grid = som.main()

noOfNeuron = 10

data = []
index = 0
for i in range(noOfNeuron):
	output = []
	for j in range(noOfNeuron):
		output.append(grid[index])
		index += 1
	data.append(output)



fig, ax = plt.subplots()
ax.matshow(data, cmap='seismic')

for (i, j), z in np.ndenumerate(data):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

plt.show()



