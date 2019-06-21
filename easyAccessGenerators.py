import numpy as np
import random
import time

#AML
#Summer 2018
#Generator functions for easyAccess.py amino acid prediction neural networks
#This file should be placed in the directory which contains the Numpy binary files generated for each protein if doing sequential fitting
#If doing fitting with crossVal text files, should be in the same directory as this file and the easyAccess.py file

#input of file names in a array for files for this dataset (should be 4 filenames or 1 filenames at a time)
def generate_arrays_from_text_files(filenames, batch_size, weight_set, dataset, long_or_short):
	print ("Starting Text Generator with files: ", filenames)
	#shuffle text files
	r_numb = int(time.time()) * 1000
	random.Random(r_numb).shuffle(filenames) #shuffle once per epoch to choose starting set of 
	file_counter = 0
	#intialize return structures
	batch_x = []
	batch_y = []
	weights = []
	#Determine if/if not using weights & which to use 
	if not weight_set == 0:
		name_weights = ""
		if weight_set == 1:
			name_weights += "sampleWeightsNonNormed"
		else:
			name_weights += "sampleWeightsNormed"
		if dataset == "30per" and long_or_short == 0:
			name_weights += "30perShort"
		if dataset == "30Per" and long_or_short == 1:
			name_weights += "30PerLong"
		classWeights = np.fromfile(name_weights, dtype = float, count = -1, sep="")
		classW = {i: classWeights[i] for i in range(0,20)}
		print ("Using weights: ", name_weights)
		print (classW)
	#Short data (each line is 293 datapoints long in file)
	if long_or_short == 0:
		while 1:
			file_raw = open(filenames[file_counter])
			file_counter = file_counter + 1
			if file_counter == len(filenames):
				file_counter = 0 
			#for every target residue in the cross val set, prepare and add to ongoing batch
			with file_raw as f:
				for line in f:
					line_list = line.split(" ")				
					list_numbers = list(map(float, line_list))				
					y = np.array(list_numbers[-20:]) #get last 20 indices from the row for the y's
					y = np.reshape(y, (20,))
					#y is a one hot check
					if not np.count_nonzero(y) == 1:
						print ("MASSIVE ERROR FOUND! SKIPPING THIS LINE")
					else:
					#weights are optional, so check if using 
						if not weight_set == 0:
							#add weight to batch weights
							weights.append(classW[np.nonzero(y)[0][0]])
						#x preparation
						x = np.array(list_numbers[:-20])
						x = np.reshape(x, (293,))
						tr = x[0:8:1]
						for i in range(0,14):
							tr = np.vstack((tr, x[0:8:1])) #should repeat the original entries for the target residue for 15 rows
						neighbor_residue_inputs = x[8:] #all of the inputs after the target residue inputs
						neighbor_reshaped = np.reshape(neighbor_residue_inputs, (15,19))
						total_x = np.hstack((tr, neighbor_reshaped))
						#add x and ys to batches
						batch_x.append(total_x)
						batch_y.append(y)
					#once enough lines stored in the batch from the cross val file, returns it
					if len(batch_y) == batch_size:
						if weight_set == 0:
							yield ([np.array(batch_x), np.array(batch_x)], np.array(batch_y))
						else:
							yield ([np.array(batch_x), np.array(batch_x)], np.array(batch_y), np.array(weights))
							weights = []
						batch_x = []
						batch_y = []
	else: #long data format (325 datapoints per a target residue)
		while 1:
			file_raw = open(filenames[file_counter])
			file_counter = file_counter + 1
			if file_counter == len(filenames):
				file_counter = 0 
			#for target residue in the cross val set, store in the batch until a batchsize is ready for return 
			with file_raw as f:
				for line in f:
					line_list = line.split(" ")
					list_numbers = list(map(float, line_list))
					#ys
					y = np.array(list_numbers[-20:]) #get last 20 indices from the row for the y's
					y = np.reshape(y, (20,))
					#check if y is one hot
					if not np.count_nonzero(y) == 1:
						print ("MASSIVE ERROR FOUND! SKIPPING THIS LINE")
					else:
						if not weight_set == 0:
							#add the weight to batch weights if needed
							weights.append(classW[np.nonzero(y)[0][0]])
						#xs
						x = np.array(list_numbers[:-20])
						x = np.reshape(x, (325,))
						tr = x[0:10:1]
						for i in range(0,14):
							tr = np.vstack((tr, x[0:10:1])) #should repeat the original entries for the target residue for 15 rows
						neighbor_residue_inputs = x[10:] #all of the inputs after the target residue inputs
						neighbor_reshaped = np.reshape(neighbor_residue_inputs, (15,21))
						total_x = np.hstack((tr, neighbor_reshaped))
						#add x and y to the batch being prepped
						batch_x.append(total_x)
						batch_y.append(y)
					#once batch size reached, yield the batch size and reset batch to empty
					if len(batch_y) == batch_size:
						if weight_set == 0:
							yield ([np.array(batch_x), np.array(batch_x)], np.array(batch_y))#, np.array(weights))
						else:
							yield ([np.array(batch_x), np.array(batch_x)], np.array(batch_y), np.array(weights))
							weights = []
						batch_x = []
						batch_y = []	

#input of list of text file containg the pdb names of the dataset
def generate_arrays_from_Protein_Files(PDB_names, batch_size, weight_set, long_or_short, binary_files):
	#shuffle PDB names
	r_numb = int(time.time()) * 1000
	random.Random(r_numb).shuffle(PDB_names) #shuffle once per epoch
	file_counter = 0
	path = binary_files #file path to the binary files
	x_batch = []
	y_batch = []
	weights = []
	#Determining the weight set to use if any
	if not weight_set == 0:
		name_weights = ""
		if weight_set == 1:
			name_weights += "sampleWeightsNonNormed"
		else:
			name_weights += "sampleWeightsNormed"
		if path == "./binaryFiles30per_SHORT/" and long_or_short == 0:
			name_weights += "30perShort"
		if path == "./binaryFiles30Per_LONG/" and long_or_short == 1:
			name_weights += "30PerLong"
		classWeights = np.fromfile(name_weights, dtype = float, count = -1, sep="")
		classW = {i: classWeights[i] for i in range(0,20)}
		print ("Using weights: ", name_weights)
		print (classW)
	#For short data (293 datapoints per target residue)
	if long_or_short == 0:
		while 1:
			#Load x and reshape
			fx = np.fromfile(path + PDB_names[file_counter] + "binaryX", dtype = float, count = -1, sep="")
			fxshape = fx.shape
			fx = np.reshape(fx, (int(fxshape[0]/293), 293)) #separated into rows - good, each row is one batch of input for a single target residue
			#load y and reshape
			fy = np.fromfile(path + PDB_names[file_counter] + "binaryY", dtype = float, count = -1, sep="")
			fyshape = fy.shape
			fy = np.reshape(fy, (int(fyshape[0]/20), 20))
			#update file counter
			file_counter = file_counter + 1
			if file_counter == len(PDB_names) - 1:
				file_counter = 0
			#create batches out of all the target residues in the binary files
			for i in range(0,fx.shape[0]):
				#ys
				y = np.copy(fy[i])
				y = np.reshape(y, (20,))
				#check y is one hot
				if not np.count_nonzero(y) == 1:
						print ("MASSIVE ERROR FOUND!")					
				else:
					if not weight_set == 0:
						#add weight to batch if using 
						weights.append(classW[np.nonzero(y)[0][0]])
					#x's
					x = np.copy(fx[i])
					x = np.reshape(x, (293,))
					#make repeated x's
					tr = x[0:8:1]
					for i in range(0,14):
						tr = np.vstack((tr, x[0:8:1])) #should repeat the original entries for the target residue for 15 rows
					neighbor_residue_inputs = x[8:] #all of the inputs after the target residue inputs
					neighbor_reshaped = np.reshape(neighbor_residue_inputs, (15,19))
					total_x = np.hstack((tr, neighbor_reshaped))
					#add x and y to the batch
					x_batch.append(total_x)
					y_batch.append(y)
				#once a batch is done, yield it
				if len(y_batch) == batch_size:
					if weight_set == 0:
						yield ([np.array(x_batch),np.array(x_batch)], np.array(y_batch))#, np.array(weights))
					else:
						yield ([np.array(x_batch),np.array(x_batch)], np.array(y_batch), np.array(weights))
						weights = []
					x_batch = []
					y_batch = []
	else: #long data (325 datapoints per target residue binary line)
		while 1:
			#load first x's and reshape
			fx = np.fromfile(path + PDB_names[file_counter] + "binaryX", dtype = float, count = -1, sep="")
			fxshape = fx.shape
			fx = np.reshape(fx, (int(fxshape[0]/325), 325)) #separated into rows - good, each row is one batch of input for a single target residue
			#load first y's and reshape
			fy = np.fromfile(path + PDB_names[file_counter] + "binaryY", dtype = float, count = -1, sep="")
			fyshape = fy.shape
			fy = np.reshape(fy, (int(fyshape[0]/20), 20))
			#check if on last file and reset file counter
			file_counter = file_counter + 1
			if file_counter == len(PDB_names) - 1:
				file_counter = 0
			#for every target residue in the binary for the protein
			for i in range(0,fx.shape[0]):
				#y
				y = np.copy(fy[i])
				y = np.reshape(y, (20,))
				#check y is one hot
				if not np.count_nonzero(y) == 1:
						print ("MASSIVE ERROR FOUND!")					
				else:
					if not weight_set == 0:
						#add weight to batch if using weights
						weights.append(classW[np.nonzero(y)[0][0]])
					#x
					x = np.copy(fx[i])
					x = np.reshape(x, (325,))
					#make repeated x's
					tr = x[0:10:1]
					for i in range(0,14):
						tr = np.vstack((tr, x[0:10:1])) #should repeat the original entries for the target residue for 15 rows
					neighbor_residue_inputs = x[10:] #all of the inputs after the target residue inputs
					neighbor_reshaped = np.reshape(neighbor_residue_inputs, (15,21))
					total_x = np.hstack((tr, neighbor_reshaped))
					#add x and y to batches being made
					x_batch.append(total_x)
					y_batch.append(y)
				#if the batch is done, yield it and reset the batch lists to empty
				if len(y_batch) == batch_size:
					if weight_set == 0:
						yield ([np.array(x_batch),np.array(x_batch)], np.array(y_batch))#, np.array(weights))
					else:
						yield ([np.array(x_batch),np.array(x_batch)], np.array(y_batch), np.array(weights))
						weights = []
					x_batch = []
					y_batch = []

#generates arrays of entire 293 or 325 datapoints in one vector for the simple feedforward dense networks
#input of file names in a array for files for this dataset (should be 4 filenames or 1 filenames at a time)
def generate_long_arrays_from_text_files(filenames, batch_size, weight_set, dataset, long_or_short):
	print ("Starting Long Text Generator with files: ", filenames)
	#shuffle list of text files
	r_numb = int(time.time()) * 1000
	random.Random(r_numb).shuffle(filenames) #shuffle once per epoch to choose starting set of 
	file_counter = 0
	batch_x = []
	batch_y = []
	weights = []
	#Get weights if using them
	if not weight_set == 0:
		name_weights = ""
		if weight_set == 1:
			name_weights += "sampleWeightsNonNormed"
		else:
			name_weights += "sampleWeightsNormed"
		if dataset == "30per" and long_or_short == 0:
			name_weights += "30perShort"
		if dataset == "30Per" and long_or_short == 1:
			name_weights += "30PerLong"
		classWeights = np.fromfile(name_weights, dtype = float, count = -1, sep="")
		classW = {i: classWeights[i] for i in range(0,20)}
		print ("Using weights: ", name_weights)
		print (classW)
	#Short data (293 long)
	if long_or_short == 0:
		while 1:
			file_raw = open(filenames[file_counter])
			file_counter = file_counter + 1
			if file_counter == len(filenames):
				file_counter = 0 
			with file_raw as f:
			#for each line in the cross val file add to the batches
				for line in f:
					#recover numpy format from text file line
					line_list = line.split(" ")				
					list_numbers = list(map(float, line_list))				
					#y
					y = np.array(list_numbers[-20:]) #get last 20 indices from the row for the y's
					y = np.reshape(y, (20,))
					#y one hot check
					if not np.count_nonzero(y) == 1:
						print ("MASSIVE ERROR FOUND! SKIPPING THIS LINE")
					else:
						if not weight_set == 0:
							#add weight to batch if using
							weights.append(classW[np.nonzero(y)[0][0]])
						#x
						x = np.array(list_numbers[:-20])
						x = np.reshape(x, (293,))
						#add x and y to batch
						batch_x.append(x)
						batch_y.append(y)
					#once a batch length is done being pulled from the cross val file, yield it and reset the batches
					if len(batch_y) == batch_size:
						if weight_set == 0:
							yield (np.array(batch_x), np.array(batch_y))#, np.array(weights))
						else:
							yield (np.array(batch_x), np.array(batch_y), np.array(weights))
							weights = []
						batch_x = []
						batch_y = []
	else:
		#for long data (325 long)
		while 1:
			file_raw = open(filenames[file_counter])
			file_counter = file_counter + 1
			if file_counter == len(filenames):
				file_counter = 0 
			with file_raw as f:
				#for every target residue in the cross val set, add to the batch
				for line in f:
					#recovering target residue line from text file format
					line_list = line.split(" ")
					list_numbers = list(map(float, line_list))
					#y
					y = np.array(list_numbers[-20:]) #get last 20 indices from the row for the y's
					y = np.reshape(y, (20,))
					#y one hot check
					if not np.count_nonzero(y) == 1:
						print ("MASSIVE ERROR FOUND! SKIPPING THIS LINE")
					else:
						if not weight_set == 0:
							#weight added to batch if using weights
							weights.append(classW[np.nonzero(y)[0][0]])
						#x
						x = np.array(list_numbers[:-20])
						x = np.reshape(x, (325,))
						#add x and y to batch
						batch_x.append(x)
						batch_y.append(y)
					#once batch is long enough, yields it and resets batch lists
					if len(batch_y) == batch_size:
						if weight_set == 0:
							yield ([np.array(batch_x), np.array(batch_x)], np.array(batch_y))
						else:
							yield ([np.array(batch_x), np.array(batch_x)], np.array(batch_y), np.array(weights))
							weights = []
						batch_x = []
						batch_y = []	

#window generator from binary files
def generate_arrays_for_window_from_protein_files(PDB_names, batch_size, weight_set, long_or_short, binary_files, even_uneven):
	#the windows are all fifteen long 
	window_size = 7
	print ("Starting generator with window size: ", str(window_size))
	#shuffling text files of PDB names
	r_numb = int(time.time()) * 1000
	random.Random(r_numb).shuffle(PDB_names) #shuffle once per epoch
	file_counter = 0
	path = binary_files #where binaries are being pulled from
	x_batch = []
	y_batch = []
	weights = []
	windows = []
	#setting weights if using them
	if not weight_set == 0:
		name_weights = ""
		if weight_set == 1:
			name_weights += "sampleWeightsNonNormed"
		else:
			name_weights += "sampleWeightsNormed"
		if binary_files == "./binaryFiles30per_SHORT/" and long_or_short == 0:
			name_weights += "30perShort"
		if binary_files == "./binaryFiles30Per_LONG/" and long_or_short == 1:
			name_weights += "30PerLong"
		classWeights = np.fromfile(name_weights, dtype = float, count = -1, sep="")
		classW = {i: classWeights[i] for i in range(0,20)}
		print ("Using weights: ", name_weights)
		print (classW)
	#for short data (only short binaries exist for window data)
	if long_or_short == 0:
		while 1:
			#even window (7 before target and 7 after)
			if even_uneven == 0:
				fx = np.fromfile(path + PDB_names[file_counter] + "evenbinaryX", dtype = float, count = -1, sep="")
			#uneven window (11 before target 2 after)
			else: 
				fx = np.fromfile(path + PDB_names[file_counter] + "unevenbinaryX", dtype = float, count = -1, sep="")
			#x
			fxshape = fx.shape
			fx = np.reshape(fx, (int(fxshape[0]/575), 575)) #separated into rows - good, each row is one batch of input for a single target residue
			#y
			fy = np.fromfile(path + PDB_names[file_counter] + "binaryY", dtype = float, count = -1, sep="")
			fyshape = fy.shape
			fy = np.reshape(fy, (int(fyshape[0]/20), 20))
			file_counter = file_counter + 1
			if file_counter == len(PDB_names):
				file_counter = 0
			#for every line in the protein binary files add to the batches
			for i in range(0,fx.shape[0]):				
				#y
				y = np.copy(fy[i])
				y = np.reshape(y, (20,))
				if not np.count_nonzero(y) == 1:
						print ("MASSIVE ERROR FOUND!")					
				else:
					if not weight_set == 0:
						#w
						weights.append(classW[np.nonzero(y)[0][0]])
					#x has both x data and the window- need to extract window and then x 
					x = np.copy(fx[i])
					x = np.reshape(x, (575,))	
					#removing window and adding to the batch			
					window = x[-282:] #the last 282 residues are the input
					windows.append(window)
					#x
					x = x[:-282]
					x = np.reshape(x, (293,))
					tr = x[0:8:1]				
					for i in range(0,14):
						tr = np.vstack((tr, x[0:8:1])) #should repeat the original entries for the target residue for 15 rows
					neighbor_residue_inputs = x[8:] #all of the inputs after the target residue inputs
					neighbor_reshaped = np.reshape(neighbor_residue_inputs, (15,19))
					total_x = np.hstack((tr, neighbor_reshaped))
					#add x and y to the batch
					y_batch.append(y)
					x_batch.append(total_x)
				#if have whole batch, yield it and reset the lists
				if len(y_batch) == batch_size:
					if weight_set == 0:
						yield ([np.array(x_batch),np.array(x_batch), np.array(windows)], np.array(y_batch))#, np.array(weights))
					else:
						yield ([np.array(x_batch),np.array(x_batch), np.array(windows)], np.array(y_batch), np.array(weights))
						weights = []
					x_batch = []
					y_batch = []
					windows =[]
	else:
		print ("ERROR!  Only 30% short binaries exist for window networks & attempted to use long!")
	
#generator for 1 shell at 30.0 angstrom and hydrophobicity added data to the 293 long 30 % short data
#for binaries
def generate_arrays_from_Protein_Files_shHydr(PDB_names, batch_size, weight_set, long_or_short, binary_files):
	#shuffling txt files of PDB names
	r_numb = int(time.time()) * 1000
	random.Random(r_numb).shuffle(PDB_names) #shuffle once per epoch
	file_counter = 0
	path = binary_files #path the binaries are in (should technically be just "./binaryFiles30Per_SHORT/"
	x_batch = []
	y_batch = []
	weights = []
	#setting sample weights if using them
	if not weight_set == 0:
		name_weights = ""
		if weight_set == 1:
			name_weights += "sampleWeightsNonNormed"
		else:
			name_weights += "sampleWeightsNormed"
		if path == "./binaryFiles30per_SHORT/" and long_or_short == 0:
			name_weights += "30perShort"
		if path == "./binaryFiles30Per_LONG/" and long_or_short == 1:
			name_weights += "30PerLong"
		classWeights = np.fromfile(name_weights, dtype = float, count = -1, sep="")
		classW = {i: classWeights[i] for i in range(0,20)}
		print ("Using weights: ", name_weights)
		print (classW)
	while 1:
		#load x and reshape
		fx = np.fromfile(path + PDB_names[file_counter] + "hydrophobicAndShellBinaryX", dtype = float, count = -1, sep="")
		fxshape = fx.shape
		fx = np.reshape(fx, (int(fxshape[0]/309), 309)) #separated into rows - good, each row is one batch of input for a single target residue
		#load y and reshape
		fy = np.fromfile(path + PDB_names[file_counter] + "binaryY", dtype = float, count = -1, sep="")
		fyshape = fy.shape
		fy = np.reshape(fy, (int(fyshape[0]/20), 20))
		file_counter = file_counter + 1
		if file_counter == len(PDB_names) - 1:
			file_counter = 0
		#add x and y to the batch for all x and y in the binaries
		for i in range(0,fx.shape[0]):
			#y
			y = np.copy(fy[i])
			y = np.reshape(y, (20,))
			if not np.count_nonzero(y) == 1:
					print ("MASSIVE ERROR FOUND!")					
			else:
				if not weight_set == 0:
					#w
					weights.append(classW[np.nonzero(y)[0][0]])
				#x
				x = np.copy(fx[i])
				x = np.reshape(x, (309,))
				#make repeated x's
				tr = x[0:9:1]
				for i in range(0,14):
					tr = np.vstack((tr, x[0:9:1])) #should repeat the original entries for the target residue for 15 rows
				neighbor_residue_inputs = x[9:] #all of the inputs after the target residue inputs
				neighbor_reshaped = np.reshape(neighbor_residue_inputs, (15,20))
				total_x = np.hstack((tr, neighbor_reshaped))
				#add x and y to the batch
				x_batch.append(total_x)
				y_batch.append(y)
			#when batch length is done, yield the batch and reset lists
			if len(y_batch) == batch_size:
				if weight_set == 0:
					yield ([np.array(x_batch),np.array(x_batch)], np.array(y_batch))
				else:
					yield ([np.array(x_batch),np.array(x_batch)], np.array(y_batch), np.array(weights))
					weights = []
				x_batch = []
				y_batch = []

#generator for 3 shell at 20.0, 30.0, 40.0 angstrom for target and same 3 shells and hydrophobicity added to each Neighbor for the 293 long 30 % short data
#for binaries
def generate_arrays_from_Protein_Files_3shHydr(PDB_names, batch_size, weight_set, long_or_short, binary_files):
#DOES BEFORE WHILE ONCE PER
	r_numb = int(time.time()) * 1000
	random.Random(r_numb).shuffle(PDB_names) #shuffle once per epoch
	file_counter = 0
	path = binary_files
	x_batch = []
	y_batch = []
	weights = []
	#set weights if using them
	if not weight_set == 0:
		name_weights = ""
		if weight_set == 1:
			name_weights += "sampleWeightsNonNormed"
		else:
			name_weights += "sampleWeightsNormed"
		if path == "./binaryFiles30per_SHORT/" and long_or_short == 0:
			name_weights += "30perShort"
		if path == "./binaryFiles30Per_LONG/" and long_or_short == 1:
			name_weights += "30PerLong"
		classWeights = np.fromfile(name_weights, dtype = float, count = -1, sep="")
		classW = {i: classWeights[i] for i in range(0,20)}
		print ("Using weights: ", name_weights)
		print (classW)
	
	while 1:
		#load x and reshape
		fx = np.fromfile(path + PDB_names[file_counter] + "hydrophobicAnd3ShellBinaryX", dtype = float, count = -1, sep="")
		fxshape = fx.shape
		fx = np.reshape(fx, (int(fxshape[0]/356), 356)) #separated into rows - good, each row is one batch of input for a single target residue
		#load y and reshape
		fy = np.fromfile(path + PDB_names[file_counter] + "binaryY", dtype = float, count = -1, sep="")
		fyshape = fy.shape
		fy = np.reshape(fy, (int(fyshape[0]/20), 20))
		#if on last file reset to first
		file_counter = file_counter + 1
		if file_counter == len(PDB_names) - 1:
			file_counter = 0
		#for each row in the binary, add x and y to the batch
		for i in range(0,fx.shape[0]):
			#y
			y = np.copy(fy[i])
			y = np.reshape(y, (20,))
			if not np.count_nonzero(y) == 1:
					print ("MASSIVE ERROR FOUND!")					
			else:
				if not weight_set == 0:
					#w
					weights.append(classW[np.nonzero(y)[0][0]])
				#x
				x = np.copy(fx[i])
				x = np.reshape(x, (356,))
				#make repeated x's
				tr = x[0:11:1]
				for i in range(0,14):
					tr = np.vstack((tr, x[0:11:1])) #should repeat the original entries for the target residue for 15 rows
				neighbor_residue_inputs = x[11:] #all of the inputs after the target residue inputs
				neighbor_reshaped = np.reshape(neighbor_residue_inputs, (15,23))
				total_x = np.hstack((tr, neighbor_reshaped))
				#add x and y to the batch
				x_batch.append(total_x)
				y_batch.append(y)
			#if batch is long enough, yield
			if len(y_batch) == batch_size:
				if weight_set == 0:
					yield ([np.array(x_batch),np.array(x_batch)], np.array(y_batch))#, np.array(weights))
				else:
					yield ([np.array(x_batch),np.array(x_batch)], np.array(y_batch), np.array(weights))
					weights = []
				x_batch = []
				y_batch = []


#generator for 3 shell at 20.0, 30.0, 40.0 angstrom for target and same 3 shells and hydrophobicity added to each Neighbor for the 293 long 30 % short data
#for binaries
def generate_arrays_from_text_files_3shHydr(filenames, batch_size, weight_set, long_or_short, binary_files):
#DOES BEFORE WHILE ONCE PER
	r_numb = int(time.time()) * 1000
	random.Random(r_numb).shuffle(filenames) #shuffle once per epoch
	file_counter = 0
	path = binary_files
	x_batch = []
	y_batch = []
	weights = []
	#set weights if using them
	if not weight_set == 0:
		name_weights = ""
		if weight_set == 1:
			name_weights += "sampleWeightsNonNormed"
		else:
			name_weights += "sampleWeightsNormed"
		name_weights += "30PerShort"
		#if path == "./binaryFiles30Per_LONG/" and long_or_short == 1:
		#	name_weights += "30PerLong"
		classWeights = np.fromfile(name_weights, dtype = float, count = -1, sep="")
		classW = {i: classWeights[i] for i in range(0,20)}
		print ("Using weights: ", name_weights)
		print (classW)
	
	while 1:

		file_raw = open(filenames[file_counter])
		file_counter = file_counter + 1
		if file_counter == len(filenames):
			file_counter = 0 
		with file_raw as f:
			#for each line in the cross val file add to the batches
			for line in f:
				#recover numpy format from text file line
				line_list = line.split(" ")				
				list_numbers = list(map(float, line_list))				
				#y
				y = np.array(list_numbers[-20:]) #get last 20 indices from the row for the y's
				y = np.reshape(y, (20,))
				#y one hot check
				if not np.count_nonzero(y) == 1:
					print ("MASSIVE ERROR FOUND! SKIPPING THIS LINE")
				else:
					if not weight_set == 0:
						#add weight to batch if using
						weights.append(classW[np.nonzero(y)[0][0]])
					#x
					x = np.array(list_numbers[:-20])
					x = np.reshape(x, (356,))
					#add x and y to batch
					#make repeated x's
					tr = x[0:11:1]
					for i in range(0,14):
						tr = np.vstack((tr, x[0:11:1])) #should repeat the original entries for the target residue for 15 rows
					neighbor_residue_inputs = x[11:] #all of the inputs after the target residue inputs
					neighbor_reshaped = np.reshape(neighbor_residue_inputs, (15,23))
					total_x = np.hstack((tr, neighbor_reshaped))
					#add x and y to the batch
					x_batch.append(total_x)
					y_batch.append(y)
					#once a batch length is done being pulled from the cross val file, yield it and reset the batches
					if len(y_batch) == batch_size:
						if weight_set == 0:
							yield ([np.array(x_batch), np.array(x_batch)], np.array(y_batch))#, np.array(weights))
						else:
							yield ([np.array(x_batch), np.array(x_batch)], np.array(y_batch), np.array(weights))
							weights = []
						x_batch = []
						y_batch = []

#generator for 3 shell at 20.0, 30.0, 40.0 angstrom for target and same 3 shells and NO hydrophobicity added to each Neighbor for the 293 long 30 % short data
#for binaries
def generate_arrays_from_text_files_3shNoHydr(filenames, batch_size, weight_set, long_or_short, binary_files):
#DOES BEFORE WHILE ONCE PER
	r_numb = int(time.time()) * 1000
	random.Random(r_numb).shuffle(filenames) #shuffle once per epoch
	file_counter = 0
	path = binary_files
	x_batch = []
	y_batch = []
	weights = []
	#set weights if using them
	if not weight_set == 0:
		name_weights = ""
		if weight_set == 1:
			name_weights += "sampleWeightsNonNormed"
		else:
			name_weights += "sampleWeightsNormed"
		name_weights += "30PerShort"
		#if path == "./binaryFiles30Per_LONG/" and long_or_short == 1:
		#	name_weights += "30PerLong"
		classWeights = np.fromfile(name_weights, dtype = float, count = -1, sep="")
		classW = {i: classWeights[i] for i in range(0,20)}
		print ("Using weights: ", name_weights)
		print (classW)
	
	while 1:

		file_raw = open(filenames[file_counter])
		file_counter = file_counter + 1
		if file_counter == len(filenames):
			file_counter = 0 
		with file_raw as f:
			#for each line in the cross val file add to the batches
			for line in f:
				#recover numpy format from text file line
				line_list = line.split(" ")				
				list_numbers = list(map(float, line_list))				
				#y
				y = np.array(list_numbers[-20:]) #get last 20 indices from the row for the y's
				y = np.reshape(y, (20,))
				#y one hot check
				if not np.count_nonzero(y) == 1:
					print ("MASSIVE ERROR FOUND! SKIPPING THIS LINE")
				else:
					if not weight_set == 0:
						#add weight to batch if using
						weights.append(classW[np.nonzero(y)[0][0]])
					#x
					x = np.array(list_numbers[:-20])
					x = np.reshape(x, (356,))
					#add x and y to batch
					#make repeated x's
					tr = x[0:11:1]
					for i in range(0,14):
						tr = np.vstack((tr, x[0:11:1])) #should repeat the original entries for the target residue for 15 rows
					neighbor_residue_inputs = x[11:] #all of the inputs after the target residue inputs
					neighbor_reshaped = np.reshape(neighbor_residue_inputs, (15,23))
					#now remove the hydrophobicity column (it is column 20)
					neighbor_reshaped = np.delete(neighbor_reshaped, -4, 1) 
					total_x = np.hstack((tr, neighbor_reshaped))
					#add x and y to the batch
					x_batch.append(total_x)
					y_batch.append(y)
					#once a batch length is done being pulled from the cross val file, yield it and reset the batches
					if len(y_batch) == batch_size:
						if weight_set == 0:
							yield ([np.array(x_batch), np.array(x_batch)], np.array(y_batch))#, np.array(weights))
						else:
							yield ([np.array(x_batch), np.array(x_batch)], np.array(y_batch), np.array(weights))
							weights = []
						x_batch = []
						y_batch = []

		

		

