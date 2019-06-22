#AML
#loads preran networks and generates confusion matrix 

#https://www.nature.com/articles/s41598-018-24760-x
#http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html
#https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix


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
import matplotlib.pyplot as plt

TF_CPP_MIN_LOG_LEVEL=2

#simplified generator functions - only using the best options found
#weights: nonormed weights

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
		print ("MASSIVE ERROR!")


#run iterative training process->create confusion matrix
def fill_conf_matrix(model, gen, n_lines):
	cm = np.zeros((20,20)) #blank cm
	#now set up to fill
	#i,j = np.unravel_index(a.argmax(), a.shape) #adapted for use from https://stackoverflow.com/questions/5469286/how-to-get-the-index-of-a-maximum-element-in-a-numpy-array-along-one-axis
	for i in range(0,n_lines):
		x_both, y = next(gen)
		#print (y)
		pred = model.predict(x_both, batch_size = 1, verbose = 0)
		y_true_index = np.unravel_index(y.argmax(), y.shape) #gets index of true label
		#print (y_true_index)
		pred_index = np.unravel_index(pred.argmax(), pred.shape) #gets index of prediction label
		#print (pred_index)
		cm[y_true_index[1]][pred_index[1]] = cm[y_true_index[1]][pred_index[1]] + 1 #increment the prediction in the cm
	return cm


def top_k(model, gen, n_lines):
	#top 1-10 accuracies
	#i,j = np.unravel_index(a.argmax(), a.shape) #adapted for use from https://stackoverflow.com/questions/5469286/how-to-get-the-index-of-a-maximum-element-in-a-numpy-array-along-one-axis
	for i in range(0,n_lines):
		x_both, y = next(gen)
		#print (y)
		
		pred = model.predict(x_both, batch_size = 1, verbose = 0)
		y_true_index = np.unravel_index(y.argmax(), y.shape) #gets index of true label
		#print (y_true_index)
		pred_index = np.unravel_index(pred.argmax(), pred.shape) #gets index of prediction label
		#print (pred_index)
		cm[y_true_index[1]][pred_index[1]] = cm[y_true_index[1]][pred_index[1]] + 1 #increment the prediction in the cm
	return cm


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
	#print ("recall")
	#print (recall)
	#print ("precision")
	#print (precision)
	return recall, precision

def heatmap(cm):
	all_els = np.sum(cm)
	#print (all_els)
	cm2 = cm/all_els #getting the heatmap vals from 0 to 1 scaled
	return cm2

def betterHeatmap(cm):
	#make elements dividied by sum of the row instead
	cm2 = np.zeros((20,20))
	for i in range(0,20):
		s_row = np.sum(cm[i])
		for j in range(0,20):
			cm2[i][j] = cm[i][j]/s_row
	return cm2


#opening DNNs
#replicated DNN
model_old = load_model("old.yaml", "old.h5", "old.png")
old_generator = generate_arrays_from_text_files(["cv2.txt"], 1)
n_lines_old =789000

#BBSeq DNN
model_seq_dep = load_model("withhydr.yaml", "withhydr.h5", "withhydr.png")
seq_dep_generator = generate_arrays_from_text_files_3shHydr(["crossValSh3HydrSet5.txt"], 1)
n_lines_cv5 = 789301
#BB DNN
model_bb = load_model("nohydr.yaml","nohydr.h5", "nohydr.png")
bb_only_gen = generate_arrays_from_text_files_3shNoHydr(["crossValSh3HydrSet2.txt"], 1)
n_lines_cv2 = 789298



#generating confusion matrix all DNN
cm_bb = fill_conf_matrix(model_bb, bb_only_gen, n_lines_cv2)
cm_bb_seq = fill_conf_matrix(model_seq_dep, seq_dep_generator, n_lines_cv5)
cm_old = fill_conf_matrix(model_old, old_generator, n_lines_old)
'''
#opening saved confusion matrices.  Note that naming got mixed up, so the binaries do not correspond with the label on the file.  Look at the end of this for the correct label to matrix IDs
cm_bb_unshaped = np.fromfile("recreationDNNConfusionMatrixNumpyBinary", dtype = float, count = -1, sep ="")
cm_bb = np.reshape(cm_bb_unshaped, (20,20))
cm_bb_seq_unshaped = np.fromfile("bbOnlyDNNConfusionMatrixNumpyBinary", dtype = float, count = -1, sep ="")
cm_bb_seq = np.reshape(cm_bb_seq_unshaped, (20,20))
cm_old_unshaped = np.fromfile("bbSeqDNNConfusionMatrixNumpyBinary", dtype = float, count = -1, sep ="")
cm_old = np.reshape(cm_old_unshaped, (20,20))
r,p = get_precision_recall_graphs(cm_bb)
r2,p2 = get_precision_recall_graphs(cm_bb_seq)
r3, p3 = get_precision_recall_graphs(cm_old)

#getting recall graphs
fig, ax = plt.subplots()
ind = np.asarray(list(range(0,40,2)))
width = 0.5
rects1 = ax.bar(ind - width, r, width, color='red')
rects2 = ax.bar(ind, r2, width, color = 'black')
rects3 = ax.bar(ind + width, r3, width, color = 'dimgrey')
ax.set_ylabel('Recall')
ax.set_title('Recall of DNNs')
ax.set_xticks(ind)
ax.set_xticklabels(('A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"))

ax.legend((rects1[0], rects2[0], rects3[0]), ('Backbone DNN', 'Backbone/Sequence DNN', 'Replicated DNN'))
plt.show()

#getting precision graphs
fig, ax = plt.subplots()
ind = np.asarray(list(range(0,40,2)))
width = 0.5
rects1 = ax.bar(ind-width, p, width, color='red')
rects2 = ax.bar(ind, p2, width, color = 'black')
rects3 = ax.bar(ind + width, p3, width, color = 'dimgrey')
ax.set_ylabel('Precision')
ax.set_title('Precision of DNNs')
ax.set_xticks(ind)
ax.set_xticklabels(('A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"))

ax.legend((rects1[0], rects2[0], rects3[0]), ('Backbone DNN', 'Backbone/Sequence DNN', 'Replicated DNN'))
plt.show()

#heatmap time!!!!
#calculate heatmaps
heatmap_bb = betterHeatmap(cm_bb)
heatmap_bb_seq = betterHeatmap(cm_bb_seq)
heatmap_recreation = betterHeatmap(cm_old)
#create graphs and heatmap
true_labels = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
pred_labels = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
fig, ax = plt.subplots()
im = ax.imshow(heatmap_bb, cmap = 'Reds')

ax.set_xticks(np.arange(20))
ax.set_yticks(np.arange(20))
ax.set_xticklabels(pred_labels)
ax.set_yticklabels(true_labels)
ax.set_xlabel("Predicted Amino Acid")
ax.set_ylabel("True Amino Acid")
cbar = ax.figure.colorbar(im, ax=ax )
#cbar.ax.set_yticklabels([ '0', '0.5', '1'])

ax.set_title("Backbone DNN Prediction Distributions ")
fig.tight_layout()
plt.show()

#second heatmap

true_labels = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
pred_labels = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
fig, ax = plt.subplots()
im = ax.imshow(heatmap_bb_seq, cmap = "Reds")

ax.set_xticks(np.arange(20))
ax.set_yticks(np.arange(20))
ax.set_xticklabels(pred_labels)
ax.set_yticklabels(true_labels)
ax.set_xlabel("Predicted Amino Acid")
ax.set_ylabel("True Amino Acid")
cbar = ax.figure.colorbar(im, ax=ax)
#cbar.ax.set_yticklabels([ '0', '0.5', '1'])

ax.set_title("Backbone/Sequence DNN Prediction Distributions")
fig.tight_layout()
plt.show()

#recreation heatmatp
true_labels = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
pred_labels = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
fig, ax = plt.subplots()
im = ax.imshow(heatmap_recreation, cmap = "Reds")

ax.set_xticks(np.arange(20))
ax.set_yticks(np.arange(20))
ax.set_xticklabels(pred_labels)
ax.set_yticklabels(true_labels)
ax.set_xlabel("Predicted Amino Acid")
ax.set_ylabel("True Amino Acid")
cbar = ax.figure.colorbar(im, ax=ax)
#cbar.ax.set_yticklabels([ '0', '0.5', '1'])

ax.set_title("Replicated DNN Prediction Distributions")
fig.tight_layout()
plt.show()
'''
#saving the confusion matrices
#xarray.tofile("./binaryFiles3/"+ name + "binaryX", sep ="")
#cm_bb.tofile("./recreationDNNConfusionMatrixNumpyBinary", sep = "")
#cm_bb_seq.tofile("./bbOnlyDNNConfusionMatrixNumpyBinary", sep = "")
#cm_old.tofile("./bbSeqDNNConfusionMatrixNumpyBinary", sep ="")
