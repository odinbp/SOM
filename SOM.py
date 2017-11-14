import math
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from SOMMnist import *
from SOMTSP import *


class SOM(object): 

	def main(self):

		exit = False

		while (exit == False):
			print("Press [1] for TSP, [2] for MNIST and [3] to exit")
			ans = int(input())

			if (ans == 3):
				print("Thanks, see you next time")
				exit = True
				break

			elif (ans == 2): 
				print("MNIST is chosen")
				print("What's your desired k?")
				k = int(input())
				sommnist = SOMMnist(n_iterations=5000,lr0 = 0.3, tlr = 2000, size0 = 6, tsize = 800, noOfNeuron = 20, k = k)
				sommnist.main()

			elif (ans == 1):
				print("TSP is chosen")
				print("What's your desired k?")
				k = int(input())
				print("What's your desired dataset [1, 2, 3, 4, 5, 6, 7, 8 , 9(new), 10(new)] ?")
				data = int(input())
				somtsp = SOMTSP(n_iterations= 3000,lr0 = 0.7, tlr = 2000, size0 = 1/10, tsize = 500, multiplier = 8, k = k, file = str(data) + '.txt') 
				somtsp.main()

			else: 
				print("Do not understand your input" )



som = SOM()
som.main()






