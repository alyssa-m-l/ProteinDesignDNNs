import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#load model(s) - most successful of the 
#1) Copy
#3) 3andHydr_BN_DROP
#5) BB_BN_DROP

from numpy import array
from keras.models import Model, model_from_yaml
from keras.layers import Dense, Activation, Input, merge, Flatten, Reshape, Multiply, Dropout, BatchNormalization, GaussianNoise, concatenate, Add
from keras.optimizers import SGD, Adam
from keras.legacy.layers import Highway
from pathlib import Path
import keras.regularizers as regularizers
from keras import backend as K
from keras.utils import plot_model
import tensorflow as tf
import matplotlib
import numpy as np
from pyrosetta import *
init()
from pyrosetta import toolbox
import matplotlib.ticker as mticks

#order in Wang et al paper (grouped by similarity of amino acid characteristics)
correct_order = ['H', 'R', 'K', 'Q', 'E', 'D', 'N', 'S', 'T', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'G', 'C', 'P']
#Order in our program, alphabetically
alphb_order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
beta = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
#For ease of interpretting the results, use these to reorder results by similar amino acids further down				



#Open binary files for a single protein file generated from the getXandYData.py or getXandYDataWithMoreFeatures.py
#input:  name of file, position to be retrieved for prediction, size parameters a/b/c (see below), and boolean switch (set to True if using  Hydrophobic data)
#output: Input for that position in the protein ready to be fed into the network, true labels from the protein's original sequence
def get_specific_position_from_Protein_File(PDB_name, position, a, b, c, booltrueifNoHydrData=False):
	#replicated: a = 293, b = 8, c = 19
	#BB/BBSeq: a =356, b = 11, c = 23 
	x_batch = []
	y_batch = []
	#X (input)
	if a == 293: #replicated DNN
		fx = np.fromfile( PDB_name + "binaryX", dtype = float, count = -1, sep="")
	else: #Modified DNNs
		fx = np.fromfile( PDB_name + "hydrophobicAnd3ShellBinaryX", dtype = float, count = -1, sep="")
	fxshape = fx.shape
	fx = np.reshape(fx, (int(fxshape[0]/a), a)) #separated into rows 
	#load y and reshape
	fy = np.fromfile(PDB_name + "binaryY", dtype = float, count = -1, sep="")
	fyshape = fy.shape
	fy = np.reshape(fy, (int(fyshape[0]/20), 20))
	print ("Correct pos: ", alphb_order[np.argmax(fy[position-1])])
	#x's
	x = np.copy(fx[position-1])
	#make repeated x's
	tr = x[0:b:1]
	for i in range(0,14):
		tr = np.vstack((tr, x[0:b:1])) #should repeat the original entries for the target residue for 15 rows
	neighbor_residue_inputs = x[b:] #all of the inputs after the target residue inputs
	neighbor_reshaped = np.reshape(neighbor_residue_inputs, (15,c))
	if booltrueifNoHydrData and a == 356:
	#	print ("stripping out hydr data")
		neighbor_reshaped = np.delete(neighbor_reshaped, -4, 1) 
	total_x = np.hstack((tr, neighbor_reshaped))
	#add x and y to the batch
	x_batch.append(total_x)
	return [np.array(x_batch),np.array(x_batch)]


#Data generator, returns batches of data to be trained until file is done 
#gets line by line data out for a single protein prediction for the replicated DNN
#input:  Name of file to be accessed, batch_size to yield
#output:  Batches of x and y data
def generate_arrays_from_Protein_Files(PDB_name, batch_size):
	x_batch = []
	y_batch = []
	#For short data (293 datapoints per target residue)
	while 1:
		#Load x and reshape
		fx = np.fromfile(path + PDB_name + "binaryX", dtype = float, count = -1, sep="")
		fxshape = fx.shape
		fx = np.reshape(fx, (int(fxshape[0]/293), 293)) #separated into rows 
		#load y and reshape
		fy = np.fromfile(path + PDB_name + "binaryY", dtype = float, count = -1, sep="")
		fyshape = fy.shape
		fy = np.reshape(fy, (int(fyshape[0]/20), 20))
		#create batches out of all the target residues in the binary files
		for i in range(0,fx.shape[0]):
			#ys
			y = np.copy(fy[i])
			y = np.reshape(y, (20,))
			#check y is one hot
			if not np.count_nonzero(y) == 1:
					print ("MASSIVE ERROR FOUND!")					
			else:
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
				yield ([np.array(x_batch),np.array(x_batch)], np.array(y_batch))#, np.array(weights))
				x_batch = []
				y_batch = []
				
				
#Data generator, returns batches of data to be trained until file is done 
#gets line by line data out for a single protein prediction for the Backbone DNN
#input:  Name of file to be accessed, batch_size to yield
#output:  Batches of x and y data		
def generate_BB_arrays_from_Protein_Files(PDB_name, batch_size):
	#gets line by line data out for a single protein prediction for the BackboneDNN
	x_batch = []
	y_batch = []
	#356 datapoints per target residue
	while 1:
		#Load x and reshape
		fx = np.fromfile(path + PDB_name + "hydrophobicAnd3ShellBinaryX", dtype = float, count = -1, sep="")
		fxshape = fx.shape
		fx = np.reshape(fx, (int(fxshape[0]/356), 356)) #separated into rows 
		#load y and reshape
		fy = np.fromfile(path + PDB_name + "binaryY", dtype = float, count = -1, sep="")
		fyshape = fy.shape
		fy = np.reshape(fy, (int(fyshape[0]/20), 20))
		#create batches out of all the target residues in the binary files
		for i in range(0,fx.shape[0]):
			#ys
			y = np.copy(fy[i])
			y = np.reshape(y, (20,))
			#check y is one hot
			if not np.count_nonzero(y) == 1:
					print ("MASSIVE ERROR FOUND!")					
			else:
				#x's
				x = np.copy(fx[i])
				x = np.reshape(x, (356,))
				#make repeated x's
				tr = x[0:11:1]
				for i in range(0,14):
					tr = np.vstack((tr, x[0:11:1])) #should repeat the original entries for the target residue for 15 rows
				neighbor_residue_inputs = x[11:] #all of the inputs after the target residue inputs
				neighbor_reshaped = np.reshape(neighbor_residue_inputs, (15,23))
				#####################
				neighbor_reshaped = np.delete(neighbor_reshaped, -4, 1) #remove hydrophobicity column
				###################
				total_x = np.hstack((tr, neighbor_reshaped))
				#add x and y to the batch
				x_batch.append(total_x)
				y_batch.append(y)
			#once a batch is done, yield it
			if len(y_batch) == batch_size:
				yield ([np.array(x_batch),np.array(x_batch)], np.array(y_batch))#, np.array(weights))
				x_batch = []
				y_batch = []
				
				
#Data generator, returns batches of data to be trained until file is done 
#gets line by line data out for a single protein prediction for the BackBoneSequence DNN
#input:  Name of file to be accessed, batch_size to yield
#output:  Batches of x and y data		
def generate_BB_Seq_arrays_from_Protein_Files(PDB_name, batch_size):
	#gets line by line data out for a single protein prediction for the BackBoneSequence DNN
	x_batch = []
	y_batch = []
	#356 datapoints per target residue
	while 1:
		#Load x and reshape
		fx = np.fromfile(path + PDB_name + "hydrophobicAnd3ShellBinaryX", dtype = float, count = -1, sep="")
		fxshape = fx.shape
		fx = np.reshape(fx, (int(fxshape[0]/356), 356)) #separated into rows 
		#load y and reshape
		fy = np.fromfile(path + PDB_name + "binaryY", dtype = float, count = -1, sep="")
		fyshape = fy.shape
		fy = np.reshape(fy, (int(fyshape[0]/20), 20))
		#create batches out of all the target residues in the binary files
		for i in range(0,fx.shape[0]):
			#ys
			y = np.copy(fy[i])
			y = np.reshape(y, (20,))
			#check y is one hot
			if not np.count_nonzero(y) == 1:
					print ("MASSIVE ERROR FOUND!")					
			else:
				#x's
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
			#once a batch is done, yield it
			if len(y_batch) == batch_size:
				yield ([np.array(x_batch),np.array(x_batch)], np.array(y_batch))#, np.array(weights))
				x_batch = []
				y_batch = []

#Generator for batches of batch_size from multiple files for the Replicated DNN
#input:  Batch_size to yield, filenames (in a list) to access
#output:  Batches fp x and y data
def generate_arrays_from_text_files(filenames, batch_size):
	print ("Starting Text Generator with files: ", filenames)
	#intialize return structures
	batch_x = []
	batch_y = []
		#Short data (each line is 293 datapoints long in file)
	while 1:
		file_raw = open(filenames[0])
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
					yield ([np.array(batch_x), np.array(batch_x)], np.array(batch_y))
					batch_x = []
					batch_y = []

#Generator for batches of batch_size from multiple files for the BackBoneSequence DNN
#input:  Batch_size to yield, filenames (in a list) to access
#output:  Batches fp x and y data
def generate_arrays_from_text_files_3shHydr(filenames, batch_size):
#DOES BE
	x_batch = []
	y_batch = []
	while 1:

		file_raw = open(filenames[0])
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
						
						yield ([np.array(x_batch), np.array(x_batch)], np.array(y_batch))#, np.array(weights))
						x_batch = []
						y_batch = []

#Generator for batches of batch_size from multiple files for the Backbone DNN
#input:  Batch_size to yield, filenames (in a list) to access
#output:  Batches fp x and y data
def generate_arrays_from_text_files_3shNoHydr(filenames, batch_size):
#DOES BEFORE WHILE ONCE PER
	x_batch = []
	y_batch = []
	while 1:

		file_raw = open(filenames[0])
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
						yield ([np.array(x_batch), np.array(x_batch)], np.array(y_batch))#, np.array(weights))
						x_batch = []
						y_batch = []



#Loads the given model
#input:  Model yaml file name, model h5 file name, name of the model if outputting image (need to uncomment plot_model line for that to happen)
#output:  keras model
def load_model(model_yaml_name, model_h5_name, graph_name):
	nnSaveModel = Path(model_yaml_name)
	if nnSaveModel.is_file():
		print ("*************Loading saved neural network model***************!!!!!!!!!!!!!!!!!!")
		yaml_file = open(model_yaml_name, 'r')
		loaded_model_yaml = yaml_file.read()
		yaml_file.close()
		model = model_from_yaml(loaded_model_yaml)
		model.load_weights(model_h5_name)
		print ("Loaded model from disk")
		lr = 0.01 #learning rate for SGD
		m = 0.9 #momentum for SGD
		nest = False
		sgd = SGD(lr = lr, momentum = m, nesterov = nest)
		model.compile(loss = 'mean_absolute_error', optimizer = sgd , metrics = ['accuracy'])
		#plot_model(model, graph_name)
		return model
	else:
		print ("MASSIVE ERROR!  Failed to load model!")


#Gets the indices for the top k elements to determine if the true labels is in the top k highest probability amino acids
#input:  array to find the top k elements of, k 
#output:  indices of k highest numbers
def get_indices_for_top_k_elements(array,k):
	#gets the indices for the top k elements in the array in largest to smallest order
	empty = []
	for i in range(1,k+1):
		index = np.where(array == np.partition(array, -1*i)[-1*i])[0][0]
		empty.append(index)
	return empty


#Carries out the top-k accuracy analysis for a model
#Input:  Model to test, generator for the model, number of lines to use in the top-k analysis
#output:  the top k-accuracy of the model for k's 1-10
def top_k(model, gen, n_lines):
	#top 1-10 accuracies
	#i,j = np.unravel_index(a.argmax(), a.shape) #adapted for use from https://stackoverflow.com/questions/5469286/how-to-get-the-index-of-a-maximum-element-in-a-numpy-array-along-one-axis
	top_ks = [0]*10
	#going to be in order top1, top2, top3, etc. up to top10 accuracy
	for i in range(0,n_lines):
		x_both, y = next(gen)
		pred = model.predict(x_both, batch_size = 1, verbose = 0)
		top_10_indices = get_indices_for_top_k_elements(pred, 10) 
		y_true_index = np.where(1==y[0])#gets index of true label
		for i in range(1,10 + 1):
			#storing number correct out of total lines for top-k accuracy 1-10
			if y_true_index in top_10_indices[:i]:
				#if it is in the first i elements of the top 10 highest value indices of the prediction values, increase number of correct predictions out of nline predictions
				top_ks[i-1] = top_ks[i-1] + 1
	top_k_accuracy = [i/n_lines for i in top_ks]
	return top_k_accuracy


correct_order = ['H', 'R', 'K', 'Q', 'E', 'D', 'N', 'S', 'T', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'G', 'C', 'P']
alphb_order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

#Changes the order of model output from alphabetic to grouped by like amino acid properties
#input of old_matrix to be reordered
def reorder_columns(old_matrix):
	#reorders numpy arrays from AA order output by network to AA order in Wang et al paper
	if old_matrix.shape == (20,20):
		blank = np.zeros(old_matrix.shape)
		#fix columns
		for i in correct_order:
			#get position in alphb order, move that array column into correct place of blank array
			a_pos = alphb_order.index(i)
			correct_pos = correct_order.index(i)
			blank[:,correct_pos] = np.copy(old_matrix[:,a_pos])
		return blank
	else:
		print ("ERROR IN REORDER COLUMNS:  Only takes 20x20 matrices and reorders the columns.  The shape given was: ", old_matrix.shape)

#For use in reorder_columns
def reorder_rows(old_matrix):
	blank_2 = np.zeros((20,20))
	#fix rows
	for i in correct_order:
		a_pos = alphb_order.index(i)
		correct_pos = correct_order.index(i)
		blank_2[correct_pos,:] = np.copy(old_matrix[a_pos,:])
	return blank_2

#for use in reorder_columns	
def reorder_columns_and_rows(old_matrix):
	reordered_by_columns = reorder_columns(old_matrix)
	complete_reorder = reorder_rows(reordered_by_columns)
	return complete_reorder
	

#Calculates precision and recall from a model confusion matrix
#input: confusion matrix to calculate precision and recall for
#output:  Recall and precision lists
def get_precision_recall_graphs(cm):
	recall = np.zeros((20))
	precision = np.zeros((20))
	for i in range(0,20):
		#for each AA in alphabetical order, calculate recall and precision
		sum_row_cm = np.sum(cm[i]) #sum of all occ of aa
		sum_col_cm = np.sum(cm[:,i]) #sum of all predictions of aa
		true_preds = cm[i][i]
		precision[i] = true_preds/sum_col_cm
		recall[i] = true_preds/sum_row_cm
	return recall, precision

#Reorder precision and recall output lists
def reorder_prec_and_recall_lists(prec_old, recall_old):
	#reorders prec and recall results to property/similarity AA order from Wang et al
	#input of two (20,) numpy arrays
	blank_prec = np.zeros(20)
	blank_recall = np.zeros(20)
	for thing in correct_order:
		correct_pos = correct_order.index(thing)
		alphb_pos = alphb_order.index(thing)
		blank_prec[correct_pos] = prec_old[alphb_pos]
		blank_recall[correct_pos] = recall_old[alphb_pos]
	return blank_recall, blank_prec

#scales confusion matrix to make a heatmap
#input: confusion matrix to scale
#output:  Scaled confusion matrix, ready to graph as heatmap
def heatmap(cm):
	all_els = np.sum(cm)
	#print (all_els)
	cm2 = cm/all_els #getting the heatmap vals from 0 to 1 scaled
	return cm2

#Better heatmap maker, instead of scaling by all inputs, scales by rows so that each row represents a distribution that sums to one
#input: confusion matrix to be scaled
#output:  Scsaled matrix
def betterHeatmap(cm):
	#make elements dividied by sum of the row instead
	cm2 = np.zeros((20,20))
	for i in range(0,20):
		s_row = np.sum(cm[i])
		for j in range(0,20):
			cm2[i][j] = cm[i][j]/s_row
	return cm2
	

#Determines if the diagonal in one confusion matrix is larger than the diagonal in the other confusion matrix
#(when using, only compare confusion matrices which were created with the same scaler
#input:  both confusion matrices to compare
def findBiggestDiagonals(cm1, cm2):
	biggest = 0
	winner = 'z'
	for i in range(0,20):
		if abs(cm1[i][i] - cm2[i][i]) > biggest:
			biggest = abs(cm1[i][i] - cm2[i][i])
			winner = correct_order[i]
	print ("Larger diagonal is: ", winner)
	print ("Difference between the two diagonals is: ", biggest)


#Creating graphics for assessing DNNs created
def create_heatmaps_precision_and_recall():
	#Trying to reorder graphics according to desired order
	cm_bb_unshaped = np.fromfile("recreationDNNConfusionMatrixNumpyBinary", dtype = float, count = -1, sep ="")
	cm_bb = np.reshape(cm_bb_unshaped, (20,20))
	cm_bb_seq_unshaped = np.fromfile("bbOnlyDNNConfusionMatrixNumpyBinary", dtype = float, count = -1, sep ="")
	cm_bb_seq = np.reshape(cm_bb_seq_unshaped, (20,20))
	cm_old_unshaped = np.fromfile("bbSeqDNNConfusionMatrixNumpyBinary", dtype = float, count = -1, sep ="")
	cm_old = np.reshape(cm_old_unshaped, (20,20))

	#reorder precision and recall arrays
	roldorder,poldorder = get_precision_recall_graphs(cm_bb)
	r,p = reorder_prec_and_recall_lists(poldorder, roldorder)
	r2old,p2old = get_precision_recall_graphs(cm_bb_seq)
	r2, p2 =  reorder_prec_and_recall_lists(p2old, r2old)
	r3old, p3old = get_precision_recall_graphs(cm_old)
	r3, p3 =  reorder_prec_and_recall_lists(p3old, r3old)

	#getting recall graphs
	fig, ax = plt.subplots()
	ind = np.asarray(list(range(0,40,2)))
	width = 0.5
	rects1 = ax.bar(ind - width, r, width, color='darkgrey')
	rects2 = ax.bar(ind, r2, width, color = 'black')
	rects3 = ax.bar(ind + width, r3, width, color = 'dimgrey')
	ax.set_ylabel('Recall')
	ax.set_title('Recall of DNNs')
	ax.set_xticks(ind)
	ax.set_xticklabels(('H', 'R', 'K', 'Q', 'E', 'D', 'N', 'S', 'T', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'G', 'C', 'P'))
	ax.legend((rects1[0], rects2[0], rects3[0]), ('Backbone DNN', 'Backbone/Sequence DNN', 'Replicated DNN'))
	plt.show()

	#getting precision graphs
	fig, ax = plt.subplots()
	ind = np.asarray(list(range(0,40,2)))
	width = 0.5
	rects1 = ax.bar(ind-width, p, width, color='darkgrey')
	rects2 = ax.bar(ind, p2, width, color = 'black')
	rects3 = ax.bar(ind + width, p3, width, color = 'dimgrey')
	ax.set_ylabel('Precision')
	ax.set_title('Precision of DNNs')
	ax.set_xticks(ind)
	ax.set_xticklabels(('H', 'R', 'K', 'Q', 'E', 'D', 'N', 'S', 'T', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'G', 'C', 'P'))
	ax.legend((rects1[0], rects2[0], rects3[0]), ('Backbone DNN', 'Backbone/Sequence DNN', 'Replicated DNN'))
	plt.show()

	#heatmaps
	#calculate heatmaps and reorder to wang et al order
	heatmap_bb_old = betterHeatmap(cm_bb)
	heatmap_bb = reorder_columns_and_rows(heatmap_bb_old)
	
	heatmap_bb_seq_old = betterHeatmap(cm_bb_seq)
	heatmap_bb_seq = reorder_columns_and_rows(heatmap_bb_seq_old)
	
	heatmap_recreation_old = betterHeatmap(cm_old)
	heatmap_recreation = reorder_columns_and_rows(heatmap_recreation_old)
	
	findBiggestDiagonals(heatmap_bb, heatmap_bb_seq)

	#create graphs and heatmap
	true_labels = ['H', 'R', 'K', 'Q', 'E', 'D', 'N', 'S', 'T', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'G', 'C', 'P']
	pred_labels = ['H', 'R', 'K', 'Q', 'E', 'D', 'N', 'S', 'T', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'G', 'C', 'P']
	fig, ax = plt.subplots()
	im = ax.imshow(heatmap_bb, cmap = 'Greys')
	ax.set_xticks(np.arange(20))
	ax.set_yticks(np.arange(20))
	ax.set_xticklabels(pred_labels)
	ax.set_yticklabels(true_labels)
	ax.set_xlabel("Predicted Amino Acid")
	ax.set_ylabel("True Amino Acid")
	cbar = ax.figure.colorbar(im, ax=ax )
	ax.set_title("Backbone DNN Prediction Distributions ")
	fig.tight_layout()
	plt.show()

	#second heatmap

	fig, ax = plt.subplots()
	im = ax.imshow(heatmap_bb_seq, cmap = "Greys")
	ax.set_xticks(np.arange(20))
	ax.set_yticks(np.arange(20))
	ax.set_xticklabels(pred_labels)
	ax.set_yticklabels(true_labels)
	ax.set_xlabel("Predicted Amino Acid")
	ax.set_ylabel("True Amino Acid")
	cbar = ax.figure.colorbar(im, ax=ax)
	ax.set_title("Backbone/Sequence DNN Prediction Distributions")
	fig.tight_layout()
	plt.show()


	#recreation heatmatp
	fig, ax = plt.subplots()
	im = ax.imshow(heatmap_recreation, cmap = "Greys")
	ax.set_xticks(np.arange(20))
	ax.set_yticks(np.arange(20))
	ax.set_xticklabels(pred_labels)
	ax.set_yticklabels(true_labels)
	ax.set_xlabel("Predicted Amino Acid")
	ax.set_ylabel("True Amino Acid")
	cbar = ax.figure.colorbar(im, ax=ax)
	ax.set_title("Replicated DNN Prediction Distributions")
	fig.tight_layout()
	plt.show()



#TODO 
def create_topk():	
	#opening DNNs
	#replicated DNN
	model_old = load_model("old.yaml", "old.h5", "old.png")
	old_generator = generate_arrays_from_text_files(["cV2.txt"], 1)
	n_lines_old =789000
	#n_lines_old = 1000

	#BBSeq DNN
	model_seq_dep = load_model("withhydr.yaml", "withhydr.h5", "withhydr.png")
	seq_dep_generator = generate_arrays_from_text_files_3shHydr(["crossValSh3HydrSet5.txt"], 1)
	n_lines_cv5 = 789301
	#n_lines_cv5 = 1000
	#BB DNN
	model_bb = load_model("nohydr.yaml","nohydr.h5", "nohydr.png")
	bb_only_gen = generate_arrays_from_text_files_3shNoHydr(["crossValSh3HydrSet2.txt"], 1)
	n_lines_cv2 = 789298
	#n_lines_cv2 = 1000

	#generating confusion matrix all DNN
	#topk_bb = top_k(model_bb, bb_only_gen, n_lines_cv2)
	#topk_seq = top_k(model_seq_dep, seq_dep_generator, n_lines_cv5)
	print("RUNNING REPLICATED")
	topk_old = top_k(model_old, old_generator, n_lines_old)
	print ("RUNNING BB")
	topk_bb = top_k(model_bb, bb_only_gen, n_lines_cv2)
	print("RUNNING BB SEQ")
	topk_bb_seq = top_k(model_seq_dep, seq_dep_generator, n_lines_cv5)
	print ("replicated top-k's: ")
	print(topk_old)
	print("BB top-k's: ")
	print(topk_bb)
	print("BB Seq top-k's: ")
	print(topk_bb_seq)

	#graphing all on same graph
	x = list(range(1,11))
	topk_bb_line , = plt.plot(x,topk_bb, color = 'darkgrey', marker = 'o', linestyle='dashed', label = "Backbone DNN")
	topk_bb_seq_line, = plt.plot(x,topk_bb_seq, color='black', marker='o', linestyle='dashed', label = 'Backbone/Sequence DNN')
	topk_old_line, = plt.plot(x,topk_old, color='dimgrey', marker ='o', linestyle='dashed', label = 'Replicated DNN')
	plt.ylabel("Top-K Accuracy")
	plt.xlabel("K")
	plt.title("Top-K Accuracy of DNNs")
	plt.legend(handles=[topk_bb_line,topk_bb_seq_line,topk_old_line])
	plt.show()


def single_pos_graph(pred_rep, pred_bb, pred_bb_seq, site_name, pro_name):
	#making bar plot of predictions for all twenty AA probs
	fig, ax = plt.subplots()
	ind = np.asarray(list(range(0,40,2)))
	width = 0.5
	dummy_prec_data = np.zeros(20) #dummy list to reuse precision and recall reorder function
	#reorder_prec_and_recall_lists(prec_old, recall_old)
	pred_rep2, dummy = reorder_prec_and_recall_lists(dummy_prec_data, pred_rep)
	print ("REPLICATED")
	true_pos = correct_order.index(site_name[0])
	mut_pos = correct_order.index(site_name[-1])
	for i in range(0, len(pred_rep2)):
		print ( pred_rep2[i])
	print ("____________________________________________________")
	top_15_pred = get_indices_for_top_k_elements(pred_rep2, 20)
	print ("top 15 in replicated: ")
	for i in range(0,len(top_15_pred)):
		if top_15_pred[i] == true_pos:
			print ("TRUE: ",i, " ", correct_order[top_15_pred[i]])
		elif top_15_pred[i] == mut_pos:
			print ("MUTANT: ", i, correct_order[top_15_pred[i]])
		else:
			#print (i, " ", correct_order[top_15_pred[i]])
			x=0
	pred_bb2, dummy = reorder_prec_and_recall_lists(dummy_prec_data, pred_bb)
	print ("BAKCBONE")
	for i in range(0, len(pred_rep2)):
		print (pred_bb2[i])
	print("____________________________________________________")
	top_15_pred_bb = get_indices_for_top_k_elements(pred_bb2, 20)
	print ("top 15 in backbone: ")
	for i in range(0,len(top_15_pred_bb)):
		if top_15_pred_bb[i] == true_pos:
			print ("TRUE: ",i, " ", correct_order[top_15_pred_bb[i]])
		elif top_15_pred_bb[i] == mut_pos:
			print ("MUTANT: ", i, correct_order[top_15_pred_bb[i]])
		else:
			#print (i, " ", correct_order[top_15_pred_bb[i]])
			x=0
		

	pred_bb_seq2, dummy = reorder_prec_and_recall_lists(dummy_prec_data, pred_bb_seq)
	print ("BBSEQ")
	for i in range(0, len(pred_rep2)):
		print (pred_bb_seq2[i])
	print("_______________________________________________")
	print (" ")
	
		
	top_15_pred_bb_seq = get_indices_for_top_k_elements(pred_bb_seq2, 20)
	print ("top 15 in backbone seq: ")
	for i in range(0,len(top_15_pred_bb_seq)):
		if top_15_pred_bb_seq[i] == true_pos:
			print ("TRUE: ",i, " ", correct_order[top_15_pred_bb_seq[i]])
		elif top_15_pred_bb_seq[i] == mut_pos:
			print ("MUTANT: ", i, correct_order[top_15_pred_bb_seq[i]])
		else:
			#print (i, " ", correct_order[top_15_pred_bb_seq[i]])
			x=0

	
	
	
	
	
	
	
	
	'''
	
	#get_indices_for_top_k_elements(array,k): (gives list of indices of top 10 preds, need to check if w.t. and mutant in there)
	
	
	#print ("Replicated: ")
	#print (pred_rep)
	#print ("BB DNN: ")
	#print (pred_bb)
	#print("BB SEQ DNN: ")
	#print (pred_bb_seq)
	rects1 = ax.bar(ind - width, pred_bb2, width, color='darkgrey') #bb
	rects2 = ax.bar(ind, pred_bb_seq2, width, color = 'black') # bb seq
	rects3 = ax.bar(ind + width, pred_rep2, width, color = 'dimgrey') # rep
	ax.set_ylabel('Amino Acid Probability at Position ' + site_name[1:len(site_name)-1])
	ax.set_title('Predicted DNN Amino Acids for '+ pro_name + ' ' + site_name[1:len(site_name) -1])
	ax.set_xticks(ind)
	ax.set_xticklabels(('H', 'R', 'K', 'Q', 'E', 'D', 'N', 'S', 'T', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'G', 'C', 'P'))
	ax.legend((rects1[0], rects2[0], rects3[0]), ('Backbone DNN', 'Backbone/Sequence DNN', 'Replicated DNN'))
	#ax.legend((rects1[0], rects3[0]), ('Backbone DNN', 'Replicated DNN'))
	#labelling true and mutant on graph

	i = 0
	#print (rects2[0])
	for i in range(0,20):
		#labelling middle bar
		rect = rects2[i]
		#rect = rects1[i]
		h1 = rects1[i].get_height()
		h2 = rects2[i].get_height()
		h3 = rects3[i].get_height()
		h = max([h1,h2,h3]) #place label at tallest height for pos
	#	h = max([h1,h3])
		if i == true_pos:
			plt.text(rect.get_x() + rect.get_width()/2.0, h, 'True', ha='center', va='bottom')
		elif i == mut_pos:
			plt.text(rect.get_x() + rect.get_width()/2.0, h, 'Exp', ha='center', va='bottom')
		i = i + 1
	plt.show()
	'''

def single_protein_predictions(name, graph_name, sites_list, site_names):
	#get_specific_position_from_Protein_File(PDB_name, position, a, b, c, booltrueifNoHydrData)
	#replicated: a = 293, b = 8, c = 19
	#BB/BBSeq: a =356, b = 11, c = 23 
	#true = BB only DNN w/ a = 356
	model_replicated = load_model("old.yaml", "old.h5", "old.png")
	model_seq_dep = load_model("withhydr.yaml", "withhydr.h5", "withhydr.png")
	model_bb = load_model("nohydr.yaml","nohydr.h5", "nohydr.png")
	sites_pred_outer = {} #key is site number, with list1 = replicated pred, list2 = bb pred, list3 = bb seq pred
	for i in range(0,len(sites_list)):
		name_current = site_names[i]
		site_number = sites_list[i]
		print ("running predictions for: ", name_current, " with site number ", site_number)
		#for each site of interest, get the prediction values for all three DNNS and store them
		#structure of output will be dictionary dict[key]: replicated_pred_list, bb_pred_list, bb_seq_pred_list
		input_replicated= get_specific_position_from_Protein_File(name, site_number, 293, 8, 19, False)
		pred_replicated = model_replicated.predict(input_replicated, batch_size = 1, verbose = 0)
		#print(pred_replicated)
		input_bb= get_specific_position_from_Protein_File(name, site_number, 356, 11, 23, True)
		pred_bb = model_bb.predict(input_bb, batch_size = 1 , verbose = 0)
		#print(pred_bb)
		input_bb_seq= get_specific_position_from_Protein_File(name, site_number, 356, 11, 23, False)
		pred_bb_seq = model_seq_dep.predict(input_bb_seq, batch_size = 1, verbose = 0)
		single_pos_graph(pred_replicated[0], pred_bb[0], pred_bb_seq[0], name_current, graph_name)
		#makeMeltTempVsPred(pred_replicated[0], pred_bb[0], pred_bb_seq[0])

def makeMeltTempVsPred(pred_rep, pred_bb, pred_bb_seq):
	#tm matthews
	#first pos is 0.0 for WT 
	#x axis data
	fig, ax = plt.subplots()
	ax.set_xlim((-15.6,0.1))
	#ax.xaxis.set_major_locator(mticks.MultipleLocator(1.0))
	meltTemps = [-8.3, 0.0, -0.2, -1.4, -7.0, -9.5, -8.0, -7.0, -7.6, -5.1, -6.4, -7.9, -8.6, -7.1, -11.5, -13.2, -12.8, -7.1, -7.7, -15.5]
	ddg = [-3.1, 0.0, 0.0, -0.3, -2.5, -3.5, -3.0, -2.6, -2.8, -2.0, -2.4, -2.9, -3.2, -2.7, -4.2, -4.7, -4.5, -2.6, -2.9, -5.5]
	correct_order_labels = ['H', 'R', 'K', 'Q', 'E', 'D', 'N', 'S', 'T', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'G', 'C', 'P']
	#y axis data
	dummy_prec_data = np.zeros(20) #dummy list to reuse precision and recall reorder function
	#reorder_prec_and_recall_lists(prec_old, recall_old)
	pred_rep2, dummy = reorder_prec_and_recall_lists(dummy_prec_data, pred_rep)
	pred_bb2, dummy = reorder_prec_and_recall_lists(dummy_prec_data, pred_bb)
	pred_bb_seq2, dummy = reorder_prec_and_recall_lists(dummy_prec_data, pred_bb_seq)
	ax.scatter(meltTemps, pred_rep2, c = 'dimgrey', label = 'Replicated DNN')
	ax.scatter(meltTemps, pred_bb2, c = 'darkgrey', label = 'Backbone DNN')
	ax.scatter(meltTemps, pred_bb_seq2, c = 'black', label = 'Backbone/Sequence DNN')
	ax.set_ylabel('Amino Acid Probability at Position 96')
	ax.set_xlabel('Delta Melting Temperature (Celsius)')
	ax.set_title('T4 Lysozyme Melting Temperatures and Predictions')
	plt.legend(loc = 'upper left')
	#labels adapted from https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
	for i, label in enumerate(correct_order_labels):
		ax.annotate(label, (meltTemps[i], pred_rep2[i]))
		ax.annotate(label, (meltTemps[i], pred_bb2[i]))
		ax.annotate(label, (meltTemps[i], pred_bb_seq2[i]))
	plt.show()
	fig, ax = plt.subplots()
	pred_rep2, dummy = reorder_prec_and_recall_lists(dummy_prec_data, pred_rep)
	pred_bb2, dummy = reorder_prec_and_recall_lists(dummy_prec_data, pred_bb)
	pred_bb_seq2, dummy = reorder_prec_and_recall_lists(dummy_prec_data, pred_bb_seq)
	ax.scatter(ddg, pred_rep2, c = 'dimgrey', label = 'Replicated DNN')
	ax.scatter(ddg, pred_bb2, c = 'darkgrey', label = 'Backbone DNN')
	ax.scatter(ddg, pred_bb_seq2, c = 'black', label = 'Backbone/Sequence DNN')
	ax.set_ylabel('Amino Acid Probability at Position 96')
	ax.set_xlabel(r'$\Delta \Delta$' + 'G (kcal/mol)')
	ax.set_title('T4 Lysozyme ' +  r'$\Delta \Delta$' +'G and Predictions')
	plt.legend(loc = 'upper left')
	#labels adapted from https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
	for i, label in enumerate(correct_order_labels):
		ax.annotate(label, (ddg[i], pred_rep2[i]))
		ax.annotate(label, (ddg[i], pred_bb2[i]))
		ax.annotate(label, (ddg[i], pred_bb_seq2[i]))
	plt.show()
	
	

def checkYs(PDB_name):
	pose = pose_from_pdb(PDB_name+".clean.pdb")
	seq = pose.sequence()
	fy = np.fromfile(PDB_name + "binaryY", dtype = float, count = -1, sep="")
	fyshape = fy.shape
	fy = np.reshape(fy, (int(fyshape[0]/20), 20))
	for i in range(int(fyshape[0]/20)):
		print ("orig seq: ", seq[i])
		print ("in fy: ", alphb_order[np.argmax(fy[i])])
		


def makeActivityBarGraph():
	activity = [100,136,18,35,6,4]
	y_pos = np.arange(len(activity))
	width = 0.8
	fig, ax = plt.subplots()
	rects1 = ax.bar(y_pos, activity, width, color = 'black')
	ax.set_ylabel('PEPX Activity (% of WT)')
	ax.set_title('PEPX Mutant Activities')
	ax.set_xticks(y_pos)
	ax.set_xticklabels(('L684H', 'F133W','Y8H', 'W161Q', 'Y258T','W425G'))
	plt.show()
	

#checkYs("Whitworth_refine_041")


#name_pro = "2LZM"
name_pro = 'Whitworth_refine_041'
#name_pro = "Q59485"
#run whits with Homology Model PEPX, Crystal Structure PEPX
#graph_name = "T4 Lysozyme"
graph_name = 'Crystal Structure PEPX'
#list_sites = [684, 133]#, 8, 161,258,425]
list_sites_updated = [685, 134]#, 9, 162,259,426]
site_names = ['L684', 'F133']#, 'Y8', 'W161','Y258','W425']
true_and_mut = ['L684H', 'F133W']#,'Y8H', 'W161Q', 'Y258T','W425G']
#single_protein_predictions(name_pro, graph_name, list_sites_updated, true_and_mut)

print ("ON CRYS STRUC")
single_protein_predictions(name_pro, graph_name, list_sites_updated, true_and_mut)
'''
graph_name = "Homology Model PEPX"
name_pro = "Q59485"
list_sites = [96]
list_sites_updated = [-1]
site_names = ["R96"]
true_and_mut = ["R96Z"]
single_protein_predictions(name_pro, graph_name, list_sites, true_and_mut)
'''

'''
#makeActivityBarGraph()
#Trying to reorder graphics according to desired order
cm_bb_unshaped = np.fromfile("recreationDNNConfusionMatrixNumpyBinary", dtype = float, count = -1, sep ="")
cm_bb = np.reshape(cm_bb_unshaped, (20,20))
cm_bb_seq_unshaped = np.fromfile("bbOnlyDNNConfusionMatrixNumpyBinary", dtype = float, count = -1, sep ="")
cm_bb_seq = np.reshape(cm_bb_seq_unshaped, (20,20))
cm_old_unshaped = np.fromfile("bbSeqDNNConfusionMatrixNumpyBinary", dtype = float, count = -1, sep ="")
cm_old = np.reshape(cm_old_unshaped, (20,20))

#heatmaps
#calculate heatmaps and reorder to wang et al order
heatmap_bb_old = betterHeatmap(cm_bb)
heatmap_bb = reorder_columns_and_rows(heatmap_bb_old)

heatmap_bb_seq_old = betterHeatmap(cm_bb_seq)
heatmap_bb_seq = reorder_columns_and_rows(heatmap_bb_seq_old)

heatmap_recreation_old = betterHeatmap(cm_old)
heatmap_recreation = reorder_columns_and_rows(heatmap_recreation_old)

print ("A and C")
findBiggestDiagonals(heatmap_recreation, heatmap_bb_seq)
print ("A and B")
findBiggestDiagonals(heatmap_bb, heatmap_recreation)
print ("B and C")
findBiggestDiagonals(heatmap_bb, heatmap_bb_seq)
'''

#pulling up cm's
