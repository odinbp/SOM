import math
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
plt.rcParams['figure.figsize'] = (11,7)

class SOMTSP(object):

	def __init__(self, n_iterations=100, lr0 = None, tlr = None, size0 = None, tsize = None, multiplier = 2, k = 200, file = None):

		self.noCities, self.cities = self.read_data(file)
		self.lr0 = float(lr0)
		self.size0 = float(self.noCities*size0*multiplier)
		self.lr = self.lr0
		self.size = self.size0
		self.tlr = tlr
		self.tsize = tsize
		self.multiplier = multiplier

		self.n_iterations = n_iterations

		self.sizeChanges = [(0,1)]
		self.lrateChanges = [(0,1)]
		self.k = k

	def init_neurons(self, d, count):
		return[[random.uniform(0.0,1.0) for i in range(d)] for j in range(count*self.multiplier)]


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
		

	def som(self, neurons, inputs, iterations):
		
		#You need only compute total distance (D) and show diagrams at every k steps
		
		for i in range(iterations+1):
			if i%self.k == 0:
				self.plot_map(inputs, neurons, i)
			self.som_one_step(neurons, inputs, i)
			
	def som_one_step(self, neurons, inputs, iter):

		#Pick a random input vector
		iv = inputs[random.randint(0,len(inputs)-1)]

		#Find winner of cometition
		winningIndex, winner = self.find_winner(iv, neurons)

		for neuronIndex, neuron in enumerate(neurons):
			d = self.circle_distance(len(neurons), neuronIndex, winningIndex)
			tn = self.neighbourhood(d, self.size)
			for i in range(len(neuron)):
				neuron[i] += self.lr*tn*(iv[i]-neuron[i])

		self.size_decay(iter)
		self.learning_decay(iter)


	def find_winner(self, iv, neurons):
		mi,mn = -100,[999999999999999.0,999999999999999.0]
		for index,neuron in enumerate(neurons):
			if self.discriminant(iv,neuron) < self.discriminant(iv,mn):
				mi = index
				mn = neuron
		return mi,mn

	def circle_distance(self, n, i, j):
		d = abs(i-j)
		return min(d, n-d)

	def size_decay(self, t):
		self.sizeChanges.append((t, math.exp(-t/self.tsize)))
		self.size = self.size0*math.exp(-t/self.tsize)
		

	def learning_decay(self, t):
		self.lrateChanges.append((t, math.exp(-t/self.tlr)))
		self.lr = self.lr0*math.exp(-t/self.tlr)

	def main(self):
		scaleFactor, scaled = self.normalize(self.cities)
		neurons = self.init_neurons(len(self.cities[0]), self.noCities)
		self.som(neurons = neurons, inputs = scaled, iterations = self.n_iterations)
		coordinates, distance = self.calculate_distance(scaled, neurons)
		distance *= scaleFactor
		plt.close()
		self.plot_finished(coordinates, distance)
		self.plot_history(self.sizeChanges, self.lrateChanges)
		return distance

	def read_data(self, file):
		cities = []
		noCities = 0
		with open('./DataSets/'+ file, 'r') as f:
			next(f)
			next(f)
			next(f)
			next(f)
			next(f)
			for line in f:
				s = line.split()
				if (s[0] == 'EOF'):
					break
				else:
					cities.append((float(s[1]),float(s[2])))
					noCities += 1
		return noCities, cities

	def normalize(self, data):
		maxX = max(d[0] for d in data)
		maxY = max(d[1] for d in data)
		scale = max(maxX, maxY)

		return scale,[(d[0]/scale, d[1]/scale) for d in data]

	def plot_map(self, inputs, neurons, iteration):		
		plt.clf()
		plt.scatter(*zip(*inputs), color='red', s=20, marker = '*')
		plt.scatter(*zip(*neurons), color='green', s=5)
		plt.plot(*zip(*(neurons+[neurons[0]])), color = 'black', alpha = 0.4) #Binding them all together
		plt.title('Iteration #{:06d}'.format(iteration) + '\n' + 'Learning rate: ' + str(round(self.lr,4)) + '\n' + "Neighbourhood size: " + str(round(self.size,4)))
		plt.draw()
		plt.pause(0.1)

	def plot_finished(self, inputs, distance):
		plt.clf()
		plt.scatter(*zip(*inputs), color='red', s=20, marker = '*')
		plt.plot(*zip(*(inputs+[inputs[0]])), color='black', alpha=0.4)
		offset = 0.01

		for i in range(0, len(inputs)):
			plt.annotate(i, xy=(inputs[i][0], inputs[i][1]+offset), fontsize=5)

		plt.title('Final path' + '\n' + 'Total distance travelled: ' + str(distance))

		#plt.show()

	def calculate_distance(self, inputs, neurons):

		match = {} #Map cities to neurons, match in dictionary
		for i,city in enumerate(inputs):
			index,neuron = self.find_winner(city, neurons)
			if index not in match:
				match[index] = [city] #put in list, can be multiple
			else: 
				match[index].append(city)

		#Find the marching order of the cities
		marching_order = []
		for j in range(len(neurons)):
			if j in match: #If neuron is associated with any city
				marching_order += match[j] #Add the cities to the marching order

		#calculate the distance of the march
		distance = 0.0
		for i in range(len(marching_order)-1): #Add distance from start to city 1, city 1 to city 2 etc. 
			distance += self.discriminant(marching_order[i], marching_order[i+1])

		distance += self.discriminant(marching_order[-1], marching_order[0]) #Add distance to get back to start

		return marching_order, distance

	def simple_plot(self, yvals,xvals=None,xtitle='X',ytitle='Y',title='Y = F(X)'):
	    xvals = xvals if xvals is not None else list(range(len(yvals)))
	    plt.plot(xvals,yvals)
	    plt.xlabel(xtitle); plt.ylabel(ytitle); plt.title(title)
	    plt.draw()

	# Each history is a list of pairs (timestamp, value).
	def plot_history(self, sizeChanges = [],lrateChanges=[],xtitle="Timestep",ytitle="Rate [Normalized to 1]",title="Change in neighbourhood and learning rate"):
		plt.figure(2)
		plt.ion()
		if len(sizeChanges) > 0:
			self.simple_plot([p[1] for p in sizeChanges], [p[0] for p in sizeChanges],xtitle=xtitle,ytitle=ytitle,title=title)
			#PLT.hold(True)
		if len(lrateChanges) > 0:
			self.simple_plot([p[1] for p in lrateChanges], [p[0] for p in lrateChanges],xtitle=xtitle,ytitle=ytitle,title=title)
		plt.ioff()
		plt.show(2)



