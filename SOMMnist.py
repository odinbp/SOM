import math
import random
import csv
import matplotlib.pyplot as plt
import mnist_basics as mb
import numpy as np
import statistics

plt.rcParams['figure.figsize'] = (11,7)

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
		iv = np.array(iv)
		weight = np.array(weight)
		dist = np.linalg.norm(iv - weight)
		return dist

	def neighbourhood(self, distance, size):
		p = 2*size**2
		if p == 0:
			return 0
		return math.exp((-distance**2)/p)
		
	def som(self, neurons, inputs, iterations, k, noOfNeuron):
		
		#You need only compute total distance (D) and show diagrams at every k steps
		#print(neurons, "som") #feil
		for i in range(iterations+1):
			print(i, "i")
			self.som_one_step(neurons, inputs, i, noOfNeuron)
			#if i%k == 0 and i != 0:
				#self.plotGrid(noOfNeuron, neurons)
			
	'''
				index = 0
				for j in range(noOfNeuron):
					output = []
					for _ in range(noOfNeuron):
						output.append(average[index])
						index += 1
					print(output)
				'''				

	def som_one_step(self, neurons, inputs, iter, noOfNeuron):

		#Pick a random input vector
		iv = inputs[random.randint(0,len(inputs)-1)]
		#Find winner of cometition
		winningIndex, winner = self.find_winner(iv[0], neurons[0])
		neurons[1][winningIndex].append(iv[1])
		iv = iv[0]
		tn = 0
		for neuronIndex, neuron in enumerate(neurons[0]):
			d = self.grid_distance(neuronIndex, winningIndex, noOfNeuron)
			tn = self.neighbourhood(d, self.size)
			#neuron = np.array(neuron)
			#ivv = np.array(iv)
			#neuron += np.subtract(self.lr*tn*ivv, self.lr*tn*neuron)
			#neuron.tolist()
			for i in range(len(neuron)):
				neuron[i] += self.lr*tn*(iv[i]-neuron[i])



		#Nødt til å teste om neurons[0] nå er lik som den var på forrige funksjon. Dårlig resultat?
		'''
		iv = np.array(iv)
		neur = np.array(neurons[0])
		neur += np.subtract(self.lr*tn*iv, self.lr*tn*neur)
		#neurons[0] = neur.tolist()
		a = neur.tolist()
		if a == neurons[0]:
			print("True")
		else:
			print("False")
		'''
		self.size_decay(iter)
		self.learning_decay(iter)

	def find_winner(self, iv, neurons):
		mi,mn = -100,[255]*784
		for index,neuron in enumerate(neurons):
			if self.discriminant(iv,neuron) < self.discriminant(iv,mn):
				mi = index
				mn = neuron
		return mi,mn

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

	def plotGrid(self, noOfNeuron, neurons):
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
			ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
		plt.ion()		
		plt.show()
		plt.draw()
		plt.pause(0.1)
		plt.close()

	def accuracy(self, neurons, data):
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
			a,b = self.find_winner(data[i][0], neurons[0])
			if average[a] == data[i][1]:
				correct += 1
		return (correct/len(data))*100


	def main(self):
		num, data = self.read_data(600)
		trainData = data[:500]
		testData = data [500:]
		scaleFactorTrain, scaledTrain = self.normalize(trainData)
		scaleFactorTest, scaledTest = self.normalize(testData)
		noOfNeuron = self.noOfNeuron
		neurons = self.init_neurons(len(trainData[0][0]), noOfNeuron)#noOfImages)
		self.som(noOfNeuron = noOfNeuron, neurons = neurons, inputs = scaledTrain, iterations = self.n_iterations, k = 500)
		train = self.accuracy(neurons, trainData)
		test = self.accuracy(neurons, testData)
		print(train, "% train", test, "% test")
		return train, test

#tlr -> learning decay, size0 -> neighbourhood, tsize - > size decay, multiplier - > size ganges med multiplier
som = SOMMnist(n_iterations=3000,lr0 = 0.1, tlr = 1000, size0 = 3, tsize = 600, noOfNeuron = 15) 
som.main()

#som = SOMMnist(n_iterations=6000,lr0 = 0.2, tlr = 1000, size0 = 3, tsize = 600, noOfNeuron = 10) 
#87.2, 79
#83.8, 70
#86.2, 69
#85.8, 71
