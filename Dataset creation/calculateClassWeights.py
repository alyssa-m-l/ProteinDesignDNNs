#AML
#6/21/18
#Opening all y's from binary, calculating the class weights for all of them

import numpy as np
import matplotlib.pyplot as plt
path = "./binaryFiles3/"

names_raw = open("finalIDsasOfJul17.txt", "r")
names = names_raw.read().splitlines()


totalN = np.zeros((20,), dtype = float)

rows = 0
for name in names:
	fy =  np.fromfile(path + name + "binaryY", dtype = float, count = -1, sep="")
	fyshape = fy.shape
	fy = np.reshape(fy, (int(fyshape[0]/20), 20))
	fy_sum = np.sum(fy, axis = 0)
	totalN = totalN + fy_sum
	rows = rows + int(fyshape[0]/20)
	print (rows)


weight_vals = np.zeros((20,), dtype = float)
weight_vals2 = np.zeros((20,), dtype = float)
for i in range(0,20):
	weight_vals[i] = float(rows/totalN[i])
	

#weight_vals.tofile("sampleWeightsNonNormed", sep = "")
weight_vals2 = np.copy(weight_vals)
weight_vals2 = weight_vals2/np.linalg.norm(weight_vals2)

#weight_vals2.tofile("sampleWeightsNormed", sep = "")

names_raw.close()
print ("Finished 30 per short")
#GENERATES GRAPHIC FOR COMPARING AA FREQ in TOTAL SET
'''
testOpen = np.fromfile("classWeightsNN", dtype = float, count =-1, sep="")
print ("shape of weights is: ", testOpen.shape)
print (testOpen)
testOpen2 = np.fromfile("classWeightsNNOverTotal", dtype = float, count =-1, sep="")
print ("shape of weights is: ", testOpen2.shape)
print (testOpen2)
'''
'''
x = np.arange(20)
labels = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


normal_freq = np.array([0.074, 0.042, 0.044, 0.059, 0.033, 0.058, 0.037, 0.074, 0.029, 0.038, 0.076, 0.072, 0.018, 0.040, 0.050, 0.081, 0.062, 0.013, 0.033, 0.068])
plt.bar(x - 0.5, normal_freq, color = 'r', align='center', width = 0.5, tick_label = labels)
plt.bar(x, testOpen,color = 'b', align = 'center', width = 0.5)
plt.title("Dataset AA Frequencies")
plt.show()
'''

