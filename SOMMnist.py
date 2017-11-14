import math
import random
import csv
import matplotlib.pyplot as plt
import mnist_basics as mb
import numpy as np
from scipy import spatial
import statistics

plt.rcParams['figure.figsize'] = (11,8)

class SOMMnist(object):

	def __init__(self, noOfNeuron, n_iterations=100, lr0 = None, tlr = None, size0 = None, tsize = None, k = 500):
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
		self.k = k

	def init_neurons(self, d, count):
		weights = [[random.uniform(0.0,1.0) for i in range(d)] for j in range(count*count)] 
		predicts = []
		for i in range(count*count):
			predicts.append([])
		return [weights, predicts]

	def discriminant(self, iv, weight):
		iv = np.array(iv)
		weight = np.array(weight)
		dist = np.linalg.norm(iv - weight)
		
		return dist

	def neighbourhood(self, distance, size):
		p = 2*size**2
		if p == 0:
			return 0
		return math.exp((-distance**2)/p)
		
	def som(self, neurons, inputs, iterations, noOfNeuron):
		
		#You need only compute total distance (D) and show diagrams at every k steps
		for i in range(iterations+1):
			self.som_one_step(neurons, inputs, i, noOfNeuron)
			if i%self.k == 0 and i != 0:
				self.plotGrid(noOfNeuron, neurons,i)	

	def som_one_step(self, neurons, inputs, iter, noOfNeuron):
		#Pick a random input vector
		iv = inputs[random.randint(0,len(inputs)-1)]
		#Find winner of cometition
		winningIndex, winner =self.determine_winner(iv[0], neurons[0], noOfNeuron)
		neurons[1][winningIndex].append(iv[1])
		iv = iv[0]

		for i in range(len(neurons[0])):
			d = self.grid_distance(i, winningIndex, noOfNeuron)
			tn = self.neighbourhood(d, self.size)
			neurons[0][i] = np.array(neurons[0][i])
			iv = np.array(iv)
			neurons[0][i] = np.add(neurons[0][i], self.lr*tn*np.subtract(iv, neurons[0][i]))

		self.size_decay(iter)
		self.learning_decay(iter)
	
	def determine_winner(self, vector, neurons, noOfNeuron):
		lowest = math.inf
		x = []
		for i in range(0, noOfNeuron*noOfNeuron, noOfNeuron):
			y = []
			for j in range(noOfNeuron):
				y.append(neurons[j+i])
			x.append(y)
		
		for index,neuron in enumerate(x):
			neuron = np.array(neuron)
			dist_list = spatial.distance.cdist(neuron, [vector])
			dist = np.min(dist_list)
			if dist < lowest:
				lowest = dist
				lowest_row = index
				lowest_col = np.argmin(dist_list)
		
		return lowest_row*noOfNeuron+lowest_col, neurons[lowest_row*noOfNeuron+lowest_col]

	def findIndex(self, index, liste):
		liste = np.array(liste)
		y, x = np.where(liste == index)
		return y[0], x[0]

	def makeList(self, noOfNeuron):
		liste = []
		for i in range(0,noOfNeuron*noOfNeuron,noOfNeuron):
			temp = []
			for j in range(noOfNeuron):
				temp.append(i+j)
			liste.append(temp)
		return liste

	def grid_distance(self, neuronIndex, winningIndex, noOfNeuron):
		deltaX = winningIndex%noOfNeuron - neuronIndex%noOfNeuron
		deltaY = math.floor(winningIndex/noOfNeuron) - math.floor(neuronIndex/noOfNeuron)
		return np.sqrt(np.square(deltaY)+np.square(deltaX))
				

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

	def plotGrid(self, noOfNeuron, neurons, iteration):
		average = []
		for z in range(len(neurons[1])):
			lengthOfList = len(neurons[1][z])
			if lengthOfList == 0:
				average.append(0)
			else:
				total = []
				for element in neurons[1][z]:
					total.append(element)
				try:
					average.append(round(statistics.mode(total)))
				except:
					average.append(round(statistics.median(total)))
		data = []
		index = 0
		grid = average
		
		for i in range(noOfNeuron):
			output = []
			for j in range(noOfNeuron):
				output.append(grid[index])
				index += 1
			data.append(output)
		
		fig, ax = plt.subplots()
		ax.matshow(data, cmap='seismic')

		for (i, j), z in np.ndenumerate(data):
			ax.text(j, i, int(z), ha='center', va='center', color='white')

		plt.title("Iteration: " + str(iteration))
		
		plt.ion()		
		plt.show()
		plt.draw()
		plt.pause(1)
		plt.close()
		plt.ioff()

	def plotFinished(self, noOfNeuron, neurons, trainAccuracy, testAccuracy):
		average = []
		for z in range(len(neurons[1])):
			lengthOfList = len(neurons[1][z])
			if lengthOfList == 0:
				average.append(0)
			else:
				total = []
				for element in neurons[1][z]:
					total.append(element)
				try:
					average.append(round(statistics.mode(total)))
				except:
					average.append(round(statistics.median(total)))
		data = []
		index = 0
		grid = average
		
		for i in range(noOfNeuron):
			output = []
			for j in range(noOfNeuron):
				output.append(grid[index])
				index += 1
			data.append(output)
		
		fig, ax = plt.subplots()
		ax.matshow(data, cmap='seismic')

		for (i, j), z in np.ndenumerate(data):
			ax.text(j, i, int(z), ha='center', va='center', color='white')
				
		plt.title('Training Finished' + '\n' + 'Training Accuracy: ' + str(trainAccuracy) +'%' +  '\n' + 'Testing Accuracy: ' + str(testAccuracy) + '%')
		plt.show()


	def accuracy(self, neurons, data, noOfNeuron):
		average = []
		for i in range(len(neurons[1])):
			lengthOfList = len(neurons[1][i])
			if lengthOfList == 0:
				average.append(0)
			else:
				total = []
				for element in neurons[1][i]:
					total.append(element)
				try:
					average.append(round(statistics.mode(total)))
				except:
					average.append(round(statistics.median(total)))

		correct = 0
		for i in range(len(data)):
			a,b = self.determine_winner(data[i][0], neurons[0], noOfNeuron)
			if average[a] == data[i][1]:
				correct += 1
		return (correct/len(data))*100

	def main(self):
		num, data = self.read_data(6000)
		trainData = data[:5400]
		testData = data [5400:]
		scaleFactorTrain, scaledTrain = self.normalize(trainData)
		scaleFactorTest, scaledTest = self.normalize(testData)
		noOfNeuron = self.noOfNeuron
		neurons = self.init_neurons(len(trainData[0][0]), noOfNeuron)#noOfImages)
		self.som(noOfNeuron = noOfNeuron, neurons = neurons, inputs = scaledTrain, iterations = self.n_iterations)
		train = self.accuracy(neurons, trainData, noOfNeuron)
		test = self.accuracy(neurons, testData, noOfNeuron)
		self.plotFinished(noOfNeuron, neurons, round(train,2), round(test,2))
		return train, test


