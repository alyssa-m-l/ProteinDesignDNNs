# Source: https://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/
#!/usr/bin/python
#AML
#Summer 2018 Research
#neural network for predicting probability of amino acids at a given position



'''
Data collection steps if training from scratch:
1) pull set of PDB id's from RCSB
2) Remove opm database proteins
3) Run getXandYAllInputs (does original paper style data (15,27) x inputs)
	OR: run window 
	OR: run shandHydr
	OR run sh3andhydr
4) Run checkBinaryExist.py with correct path and filenames you want
5) If doing cross validation, run createCrossVal to create text files of the proteins and lines which will go in each cross val set
	5b) then run cross val again for each created set 
6) Check the file names in parameter function here
7) Set parameters & check you won't overwrite a old model first in file explorer when the name is generated
8) Run the training
	- if doing a set number of batches, set end_sims and make sure to calculate the right number_final_validations_for_end_sims so that the evaluate_generator covers the entire 
	evaluation data set at the end for cross validation
9) run a separate model for each cross validation data set
10) put the generated cross validation acc and loss files together to get the cross validation mean acc and acc std dev for your model performance
11) For future use of the trained network, select the network which did the best on its final cross validation accuracy from the evaluate generator

'''

#Best working model to remake the paper: "m" with nest= False, using batch_fit with 360 for number_final_validations_for_end_sims and 20000 for end_sims


from easyAccessGenerators import *
from time import time
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
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt


TF_CPP_MIN_LOG_LEVEL=2
#########################################################################################################
# code to allow hault training by pressing esc key and pressing enter 
import sys, termios, atexit
from select import select

# save the terminal settings
fd = sys.stdin.fileno()
new_term = termios.tcgetattr(fd)
old_term = termios.tcgetattr(fd)

# new terminal setting unbuffered
new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)

# switch to normal terminal
def set_normal_term():
    termios.tcsetattr(fd, termios.TCSAFLUSH, old_term)

# switch to unbuffered terminal
def set_curses_term():
    termios.tcsetattr(fd, termios.TCSAFLUSH, new_term)

def putch(ch):
    sys.stdout.write(ch)

def getch():
    return sys.stdin.read(1)

def getche():
    ch = getch()
    putch(ch)
    return ch

def kbhit():
    dr,dw,de = select([sys.stdin], [], [], 0)
    return (dr != [])
  

#########################################################################################################


#########################################################################################################
##
##												READ DATA
##
#########################################################################################################


#########################################################################################################
##
##												NETWORK 
##
#########################################################################################################
# use this link to learn how to combine networks 
# https://statcompute.wordpress.com/2017/01/08/an-example-of-merge-layer-in-keras/

#main organizer function
def set_parameters_and_run():
	# ##############################   model set up parameters ################
	model_summary = True #show model summary in terminal
	model_summary_image = False #generate model structure .svg image
	dataset = "30Per" #Datasets 30% seq identity: 30Per, 90% seq identity
	lr = 0.01 #learning rate for SGD
	m = 0.9 #momentum for SGD
	nest = False #Set Nesterov momentum state for SGD (true= uses Nest Momentum, False is does not)
	modsTag = "cv5dp0_05" #if using a different model structure, mark so here (ex: batchNormed, Dropout0_20, etc.)
	mType = "allbatchNorm3ShHydr" #model type:  Determines what model to generate from the model functions
	even_uneven = 0 #If using window model, 0 is even windows (7 residues, target, 7 residues), 1 is uneven (11 residues N term side, target, 2 residues C term side)
	tType = "normal" #Training type: "normal" uses fit_generator method for training, "batch" used fit_batch method
	if mType == "d": #catch for dense models
		print ("ONLY SUPPORTED DENSE is 90PER DATASETS")
		print ("Autosetting dataset to 90 %")
		dataset = "90Per"
	###############################   Models Implemented So Far ###################
	#Models separated by needed generator types
	original_models = ["m","mpr", "mstf", "combined", "bsf", "mstf", "mpr", "allmBN", "allmBNDrop"] #models using from text_files generator
	#NOTE: "m" can do sequential by setting sequential to true, does even or odd windows also by setting correctly
	window_models =  ["w",  "wd", "ws",  "wsd"] #uses window from protein files generator
	simple_dense_models =["d"] #uses the simple dense network generator
	sh3andHydr_models = ["m3ShHydr", "allbatchNorm3ShHydr","inputbatchNorm3ShHydr", "subnetworksbatchNorm3ShHydr", "allbatchNorm3ShHydrDrop"] #uses the ShHydr Generator
	shandHydr_models = ["mShHydr", "combinedShHydr", "allbatchNormShHydr" , "inputbatchNormShHydr", "subnetworksbatchNormShHydr", "allL2ShHydr", "allbatchNormShHydrDrop"] #uses the 3ShHydr generator
	################	
	
	###############################   Training and Validation Parameters ####################
	continuous_training = False #determines if will train continuously #NOTE: batch only does false option
	weight_set = 1 #0: no weights, 1: nonNormed Weights, 3: Normed Weights
	sequential = False
	crossValSet = 5 #1,2,3,4,5 - controls which cross val set is held out
	fit_batch_size = 8000 #size of batch for fitting generator
	validate_batch_size = 1000 #size of batch for validation from validation generator
	save_data_every_x_epochs = 25 #Updates graphs
	long_or_short = 0 #0: short (15,27 input), 1: long (15,31 input) for 30Per Dataset
	graph_acc = True #Controls if intermediate Acc graphs will be generated True: yes, False: no
	graph_loss = False #Controls if intermediate loss graphs will be generated True: yes, False: no
	binary_file = "/binaryFiles30Per_SHORT/" #Controls what file binary inputs will be searched for in.  #NOTE: only important with sequential training
	end_sims = 1000 #Controls number of epochs to train for on non-continuous training
	number_final_validations_for_end_sims = 87 #number of batches to yield from generator to cover entire dataset for cross validation statistics
	lmbda = 0.001 #Lambda for l2 regularization if it is present in mType (default value is 0.001)  
	dp = 0.05 #Drop Probability if present in mType (default value is 0.20)
	#####
	
	################################     Preparing Datasets  ########################
	if sequential: 
		splitting_percentage = 0.80 #80-20 validation split with Protein RCSB IDs
		if dataset == "30Per":
			#30% Data
			if long_or_short == 0:
			#will pull data from 30Per_SHORT file
				if mType in sh3andHydr_models:
					all_ids_file = "3shellsAndHydrBinaries2.txt"
					binary_file = "./binaryFiles30Per_SHORT/"
				elif mType in shandHydr_models:
					all_ids_file = "shellAndHydrBinariesInitial2.txt"
					binary_file = "./binaryFiles30Per_SHORT/"
				elif not (mType in window_models):
					all_ids_file = "finalIDs30PerShort.txt"
					binary_file = "./binaryFiles30Per_SHORT/"
				else:
				#window mType
					if even_uneven == 0:
						all_ids_file = "evenWindowExists30Short.txt"
						binary_file = "./binaryFiles30Per_SHORT/"
					else:
						all_ids_file = "unevenWindowExists30Short.txt"
						binary_file = "./binaryFiles30Per_SHORT/"
			else:
			#30Per_LONG binaries
				all_ids_file = "finalIDs30PerLong.txt"
				binary_file = "./binaryFiles30Per_LONG/"
		else:
		#90 Per binaries
			all_ids_file = "finalIDsasOfJul18.txt"
			binary_file = "./binaryFiles3/"
		
		print ("USING FILES: ", all_ids_file)
		all_ids_raw = open(all_ids_file, "r")
		all_ids = all_ids_raw.read().splitlines()
		#splitting into 80% training, 20% validation manually
		index_split_at = int(splitting_percentage*len(all_ids))
		training_files = all_ids[:index_split_at] #80%
		validation_files = all_ids[index_split_at:] #20%
	else:
		#using a cross validation set
		if dataset == "90Per":
			if crossValSet == 1:
				print ("___________________cv1 held out as test set___________________-")
				training_files = ["cv3.txt", "cv4.txt", "cv2.txt", "cv5.txt"]
				validation_files = ["cv1.txt"]
		
			if crossValSet == 2:
				print ("cv2 held out as test set")
				training_files = ["cv3.txt", "cv4.txt", "cv1.txt", "cv5.txt"]
				validation_files = ["cv2.txt"]
			if crossValSet == 3:
				print ("cv3 held out as test set")
				training_files = ["cv1.txt", "cv4.txt", "cv2.txt", "cv5.txt"]
				validation_files = ["cv3.txt"]
			if crossValSet == 4:
				print ("cv4 held out as test set")
				training_files = ["cv3.txt", "cv1.txt", "cv2.txt", "cv5.txt"]
				validation_files = ["cv4.txt"]
			if crossValSet == 5:
				print ("cv5 held out as test set")
				training_files = ["cv3.txt", "cv4.txt", "cv2.txt", "cv1.txt"]
				validation_files = ["cv5.txt"]
			
		else:
			if long_or_short == 0 and (not mType in sh3andHydr_models) and (not mType in shandHydr_models):
				if crossValSet == 1:
					print ("___________________cv1 30 per short held out as test set___________________-")
					training_files = ["cv330PerShort.txt", "cv430PerShort.txt", "cv230PerShort.txt", "cv530PerShort.txt"]
					validation_files = ["cv130PerShort.txt"]
				
				if crossValSet == 2:
					print ("cv2 held out as test set")
					training_files = ["cv330PerShort.txt", "cv430PerShort.txt", "cv130PerShort.txt", "cv530PerShort.txt"]
					validation_files = ["cv230PerShort.txt"]
				if crossValSet == 3:
					print ("cv3 held out as test set")
					training_files = ["cv130PerShort.txt", "cv430PerShort.txt", "cv230PerShort.txt", "cv530PerShort.txt"]
					validation_files = ["cv330PerShort.txt"]
				if crossValSet == 4:
					print ("cv4 held out as test set")
					training_files = ["cv330PerShort.txt", "cv130PerShort.txt", "cv230PerShort.txt", "cv530PerShort.txt"]
					validation_files = ["cv430PerShort.txt"]
				if crossValSet == 5:
					print ("cv5 held out as test set")
					training_files = ["cv330PerShort.txt", "cv430PerShort.txt", "cv230PerShort.txt", "cv130PerShort.txt"]
					validation_files = ["cv530PerShort.txt"]
				
			elif (not mType in sh3andHydr_models) and (not mType in shandHydr_models):
				if crossValSet == 1:
					print ("___________________cv1 30 per long held out as test set___________________-")
					training_files = ["cv330PerLong.txt", "cv430PerLong.txt", "cv230PerLong.txt", "cv530PerLong.txt"]
					validation_files = ["cv130PerLong.txt"]
				
				if crossValSet == 2:
					print ("cv2 held out as test set")
					training_files = ["cv330PerLong.txt", "cv430PerLong.txt", "cv130PerLong.txt", "cv530PerLong.txt"]
					validation_files = ["cv230PerLong.txt"]
				if crossValSet == 3:
					print ("cv3 held out as test set")
					training_files = ["cv130PerLong.txt", "cv430PerLong.txt", "cv230PerLong.txt", "cv530PerLong.txt"]
					validation_files = ["cv330PerLong.txt"]
				if crossValSet == 4:
					print ("cv4 held out as test set")
					training_files = ["cv330PerLong.txt", "cv130PerLong.txt", "cv230PerLong.txt", "cv530PerLong.txt"]
					validation_files = ["cv430PerLong.txt"]
				if crossValSet == 5:
					print ("cv5 held out as test set")
					training_files = ["cv330PerLong.txt", "cv430PerLong.txt", "cv230PerLong.txt", "cv130PerLong.txt"]
					validation_files = ["cv530PerLong.txt"]
					
			elif mType in sh3andHydr_models:
				if crossValSet == 1:
					print ("___________________cv1 held out as test set___________________-")
					training_files = ["crossValSh3HydrSet2.txt", "crossValSh3HydrSet3.txt", "crossValSh3HydrSet4.txt", "crossValSh3HydrSet5.txt"]
					validation_files = ["crossValSh3HydrSet1.txt"]
				if crossValSet == 2:
					print ("___________________cv2 held out as test set___________________-")
					training_files = ["crossValSh3HydrSet1.txt", "crossValSh3HydrSet3.txt", "crossValSh3HydrSet4.txt", "crossValSh3HydrSet5.txt"]
					validation_files = ["crossValSh3HydrSet2.txt"]

				if crossValSet == 3:
					print ("___________________cv3 held out as test set___________________-")
					training_files = ["crossValSh3HydrSet1.txt", "crossValSh3HydrSet2.txt", "crossValSh3HydrSet4.txt", "crossValSh3HydrSet5.txt"]
					validation_files = ["crossValSh3HydrSet3.txt"]
				
				if crossValSet == 4:
					print ("___________________cv4 held out as test set___________________-")
					training_files = ["crossValSh3HydrSet2.txt", "crossValSh3HydrSet3.txt", "crossValSh3HydrSet1.txt", "crossValSh3HydrSet5.txt"]
					validation_files = ["crossValSh3HydrSet4.txt"]
					
				if crossValSet == 5:				
					print ("___________________cv5 held out as test set___________________-")
					training_files = ["crossValSh3HydrSet2.txt", "crossValSh3HydrSet3.txt", "crossValSh3HydrSet4.txt", "crossValSh3HydrSet1.txt"]
					validation_files = ["crossValSh3HydrSet5.txt"]
					
	###############
	
	#########################     Model names   ######################
	#Create basename for all network files from used parameters
	basename = ""
	basename += mType
	if long_or_short == 0:
		basename += "short"
	else:
		basename += "long"
	basename += dataset
	basename += "train" + str(fit_batch_size) + "val" + str(validate_batch_size)
	if weight_set == 0:
		basename += "NoWeights"
	elif weight_set == 1:
		basename += "NonNormedWeights"
	elif weight_set == 2:
		basename += "NormedWeights"
	if nest:
		basename += "NestTrue"
	else:
		basename += "NestFalse"
	if sequential:
		basename += "Seq"
	else:
		basename += "Cv"
	basename += modsTag
	#creating actual names
	print ("Model name is : ", basename)
	model_yaml_name = basename +".yaml"
	model_h5_name = basename +".h5"
	####
	
	pause = input("keep going?")
	
	##########################   Creating Model   #########################
	# WILL load .yaml and .h5 if they exist automatically and resume training at this point!!!!!
	nnSaveModel = Path(model_yaml_name)
	if nnSaveModel.is_file():
		print ("*************Loading saved neural network model***************!!!!!!!!!!!!!!!!!!")
		yaml_file = open(model_yaml_name, 'r')
		loaded_model_yaml = yaml_file.read()
		yaml_file.close()
		model = model_from_yaml(loaded_model_yaml)
		model.load_weights(model_h5_name)
		print ("Loaded model from disk")
		sgd = SGD(lr = lr, momentum = m, nesterov = nest)
		model.compile(loss = 'mean_absolute_error', optimizer = sgd , metrics = ['accuracy'])
	else:
		print ("CREATING NEW LOGS!!!!!!!!!")
		store_acc = open(basename + "ValidationLog.txt", "w")
		store_acc.close()
		store_loss = open(basename + "LossLog.txt", "w")
		store_loss.close()
		if mType == "m":
			print ("making merge model")
			if long_or_short == 0:
				model = createNN(lr, m, nest, model_summary, model_summary_image )
			else:
				model = createNNLongInput(lr, m, nest, model_summary, model_summary_image )
		elif mType == "d":
			print ("making dense model")
			model = createDenseNN(lr, m, nest, model_summary, model_summary_image )
		elif mType == "bsf":
			model = create_BSF_NN(lr, m, nest, model_summary, model_summary_image, lmbda, dp)
		elif mType == "mstf":
			model = createNNSkipToFlatten(lr, m, nest, model_summary, model_summary_image)
		elif mType == "wd":
			model = createWindowNN_batchNormDropout(lr, m, nest, model_summary, model_summary_image, dp)
		elif mType == "mpr":
			model = createNNPsuedoResidual(lr, m, nest, model_summary, model_summary_image )
		elif mType == "ws":
			model = createWindowSmallNN(lr, m, nest, model_summary, model_summary_image )
		elif mType == "wsd":
			model = createWindowSmallDropNN(lr, m, nest, model_summary, model_summary_image, dp)
		elif mType == "w":
			model = createWindowNN(lr, m, nest, model_summary, model_summary_image )
		elif mType == "mShHydr":
			model = shellAndHydrNN(lr, m, nest, model_summary, model_summary_image )
		elif mType == "combinedShHydr":
			model = create_BSF_NN_skip_bn_ShandHydr(lr, m, nest, model_summary, model_summary_image, lmbda, dp ) 
		elif mType == "allbatchNormShHydr":
			model = createBNShHydrAll(lr, m, nest, model_summary, model_summary_image)
		elif mType == "allbatchNormShHydrDrop":
			model = model = createBNShHydrAllDropout(lr, m, nest, model_summary, model_summary_image, dp)
		elif mType == "inputbatchNormShHydr":
			model = createBNShHydrInput(lr, m, nest, model_summary, model_summary_image)
		elif mType == "subnetworksbatchNormShHydr":
			model = createBNShHydrSubnetworks(lr, m, nest, model_summary, model_summary_image)
		elif mType == "combined":
			model = create_BSF_NN_skip_bn(lr, m, nest, model_summary, model_summary_image, lmbda, dp )
		elif mType == "m3ShHydr":
			model = createNN3ShellAndHydr(lr, m, nest, model_summary, model_summary_image)
		elif mType == "allbatchNorm3ShHydr":
			model = model =  createBN3ShHydrAll(lr, m, nest, model_summary, model_summary_image)
		elif mType == "allbatchNorm3ShHydrDrop":
			model = model = createBN3ShHydrAllDropout(lr, m, nest, model_summary, model_summary_image, dp)
		elif mType == "inputbatchNorm3ShHydr":
			model = model =  createBN3ShHydrInput(lr, m, nest, model_summary, model_summary_image)
		elif mType == "subnetworksbatchNorm3ShHydr":
			model = model =  createBN3ShHydrSubnetworks(lr, m, nest, model_summary, model_summary_image)
		elif mType == "allmBN":
			model = createBNmAll(lr, m, nest, model_summary, model_summary_image)
		elif mType == "allmBNDrop":
			model = createBNmDropout(lr, m, nest, model_summary, model_summary_image, dp)
		elif mType == "allL2ShHydr":
			model = createNNL2ShandHydr(lr, m, nest, model_summary, model_summary_image, dp)
	#####
	#                    Training Model
	if tType == "normal":	
		training(model, training_files, validation_files, fit_batch_size, validate_batch_size, weight_set, long_or_short, binary_file, mType, save_data_every_x_epochs, basename, continuous_training, graph_acc, graph_loss, sequential, end_sims, dataset, even_uneven, original_models,window_models, simple_dense_models, sh3andHydr_models, shandHydr_models, number_final_validations_for_end_sims )
	else:
		batchTraining(model, training_files, validation_files, fit_batch_size, validate_batch_size, weight_set, long_or_short, binary_file, mType, save_data_every_x_epochs, basename, continuous_training, graph_acc, graph_loss, sequential, end_sims, dataset, even_uneven, original_models,window_models, simple_dense_models, sh3andHydr_models, shandHydr_models, number_final_validations_for_end_sims)
	#####
	#                     Saving Model
	save_model(model, model_yaml_name, model_h5_name)
	
	

###################################### Paper Direct Recreation ##################################################
def createNN(lr, m, nest, showsumm, showimage ):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,27), name = "ResidueSubnetworkInput")
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,27))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu')(hidden_r1)
	hidden_r3 = Dense(100, activation = 'relu')(hidden_r2)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,27), name = "WeightSubNetworkInput")
	hidden_w1 = Dense(100, activation = 'relu')(weightNN)
	hidden_w2 = Dense(100, activation = 'relu')(hidden_w1)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(hidden_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
			
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	hidden_m1 = Dense(300, activation = 'relu')(flat_via_reshape)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel
	
	
def createBNmDropout(lr, m, nest, showsumm, showimage, dp):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,27), name = "ResidueSubnetworkInput")
	norminput_p = BatchNormalization()(residueProbNN)
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,27))(norminput_p)
	norm_r1 = BatchNormalization()(hidden_r1)
	drop_r1 = Dropout(dp)(norm_r1)
	hidden_r2 = Dense(100, activation = 'relu')(drop_r1)
	norm_r2 = BatchNormalization()(hidden_r2)
	drop_r2 = Dropout(dp)(norm_r2)
	hidden_r3 = Dense(100, activation = 'relu')(drop_r2)
	norm_r3 = BatchNormalization()(hidden_r3)
	drop_r3 = Dropout(dp)(norm_r3)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(norm_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()
		
	#WEIGHT NETWORK
	weightNN = Input(shape = (15,27), name = "WeightSubNetworkInput")
	norminput_w = BatchNormalization()(weightNN)
	hidden_w1 = Dense(100, activation = 'relu')(norminput_w)
	norm_w1 = BatchNormalization()(hidden_w1)
	drop_w1 = Dropout(dp)(norm_w1)
	hidden_w2 = Dense(100, activation = 'relu')(drop_w1)
	norm_w2 = BatchNormalization()(hidden_w2)
	drop_w2 = Dropout(dp)(norm_w2)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(drop_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
	
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	norm_reshape = BatchNormalization()(flat_via_reshape)
	hidden_m1 = Dense(300, activation = 'relu')(norm_reshape)
	norm_m1 = BatchNormalization()(hidden_m1)
	drop_m1 = Dropout(dp)(norm_m1)
	hidden_m2 = Dense(100, activation = 'relu')(drop_m1)
	norm_m2 = BatchNormalization()(hidden_m2)
	drop_m2 = Dropout(dp)(norm_m2)
	hidden_m3 = Dense(100, activation = 'relu')(drop_m2)
	norm_m3 = BatchNormalization()(hidden_m3)
	drop_m3 = Dropout(dp)(norm_m3)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(drop_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel


def createBNmAll(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,27), name = "ResidueSubnetworkInput")
	norminput_p = BatchNormalization()(residueProbNN)
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,27))(norminput_p)
	norm_r1 = BatchNormalization()(hidden_r1)
	hidden_r2 = Dense(100, activation = 'relu')(norm_r1)
	norm_r2 = BatchNormalization()(hidden_r2)
	hidden_r3 = Dense(100, activation = 'relu')(norm_r2)
	norm_r3 = BatchNormalization()(hidden_r3)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(norm_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()
	
	#WEIGHT NETWORK
	weightNN = Input(shape = (15,27), name = "WeightSubNetworkInput")
	norminput_w = BatchNormalization()(weightNN)
	hidden_w1 = Dense(100, activation = 'relu')(norminput_w)
	norm_w1 = BatchNormalization()(hidden_w1)
	hidden_w2 = Dense(100, activation = 'relu')(norm_w1)
	norm_w2 = BatchNormalization()(hidden_w2)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(norm_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
	
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	norm_reshape = BatchNormalization()(flat_via_reshape)
	hidden_m1 = Dense(300, activation = 'relu')(norm_reshape)
	norm_m1 = BatchNormalization()(hidden_m1)
	hidden_m2 = Dense(100, activation = 'relu')(norm_m1)
	norm_m2 = BatchNormalization()(hidden_m2)
	hidden_m3 = Dense(100, activation = 'relu')(norm_m2)
	norm_m3 = BatchNormalization()(hidden_m3)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(norm_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel


############################################## Shell and Hydrophobic Networks #########################################

##################### 1 shell at 30.0 on target residue and 1 hydrophobic per Neighbor data points added ##############
def createNNL2ShandHydr(lr, m, nest, showsumm, showimage, lmda):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,29), name = "ResidueSubnetworkInput")
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,29), kernel_regularizer = regularizers.l2(lmda))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_r1)
	hidden_r3 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_r2)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,29), name = "WeightSubNetworkInput")
	hidden_w1 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(weightNN)
	hidden_w2 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_w1)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(hidden_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
			
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	hidden_m1 = Dense(300, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(flat_via_reshape)
	hidden_m2 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel


def createBNShHydrAll(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,29), name = "ResidueSubnetworkInput")
	norminput_p = BatchNormalization()(residueProbNN)
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,29))(norminput_p)
	norm_r1 = BatchNormalization()(hidden_r1)
	hidden_r2 = Dense(100, activation = 'relu')(norm_r1)
	norm_r2 = BatchNormalization()(hidden_r2)
	hidden_r3 = Dense(100, activation = 'relu')(norm_r2)
	norm_r3 = BatchNormalization()(hidden_r3)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(norm_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()
	
	#WEIGHT NETWORK
	weightNN = Input(shape = (15,29), name = "WeightSubNetworkInput")
	norminput_w = BatchNormalization()(weightNN)
	hidden_w1 = Dense(100, activation = 'relu')(norminput_w)
	norm_w1 = BatchNormalization()(hidden_w1)
	hidden_w2 = Dense(100, activation = 'relu')(norm_w1)
	norm_w2 = BatchNormalization()(hidden_w2)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(norm_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
	
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	norm_reshape = BatchNormalization()(flat_via_reshape)
	hidden_m1 = Dense(300, activation = 'relu')(norm_reshape)
	norm_m1 = BatchNormalization()(hidden_m1)
	hidden_m2 = Dense(100, activation = 'relu')(norm_m1)
	norm_m2 = BatchNormalization()(hidden_m2)
	hidden_m3 = Dense(100, activation = 'relu')(norm_m2)
	norm_m3 = BatchNormalization()(hidden_m3)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(norm_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel
	
	
def createBNShHydrAllDropout(lr, m, nest, showsumm, showimage, dp):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,29), name = "ResidueSubnetworkInput")
	norminput_p = BatchNormalization()(residueProbNN)
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,29))(norminput_p)
	norm_r1 = BatchNormalization()(hidden_r1)
	drop_r1 = Dropout(dp)(norm_r1)
	hidden_r2 = Dense(100, activation = 'relu')(drop_r1)
	norm_r2 = BatchNormalization()(hidden_r2)
	drop_r2 = Dropout(dp)(norm_r2)
	hidden_r3 = Dense(100, activation = 'relu')(drop_r2)
	norm_r3 = BatchNormalization()(hidden_r3)
	drop_r3 = Dropout(dp)(norm_r3)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(norm_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()
	
	#WEIGHT NETWORK
	weightNN = Input(shape = (15,29), name = "WeightSubNetworkInput")
	norminput_w = BatchNormalization()(weightNN)
	hidden_w1 = Dense(100, activation = 'relu')(norminput_w)
	norm_w1 = BatchNormalization()(hidden_w1)
	drop_w1 = Dropout(dp)(norm_w1)
	hidden_w2 = Dense(100, activation = 'relu')(drop_w1)
	norm_w2 = BatchNormalization()(hidden_w2)
	drop_w2 = Dropout(dp)(norm_w2)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(drop_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
	
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	norm_reshape = BatchNormalization()(flat_via_reshape)
	hidden_m1 = Dense(300, activation = 'relu')(norm_reshape)
	norm_m1 = BatchNormalization()(hidden_m1)
	drop_m1 = Dropout(dp)(norm_m1)
	hidden_m2 = Dense(100, activation = 'relu')(drop_m1)
	norm_m2 = BatchNormalization()(hidden_m2)
	drop_m2 = Dropout(dp)(norm_m2)
	hidden_m3 = Dense(100, activation = 'relu')(drop_m2)
	norm_m3 = BatchNormalization()(hidden_m3)
	drop_m3 = Dropout(dp)(norm_m3)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(drop_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel


def createBNShHydrInputs(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,29), name = "ResidueSubnetworkInput")
	norminput_p = BatchNormalization()(residueProbNN)
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,29))(norminput_p)
	norm_r1 = BatchNormalization()(hidden_r1)
	hidden_r2 = Dense(100, activation = 'relu')(norm_r1)
	hidden_r3 = Dense(100, activation = 'relu')(hidden_r2)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()
	
	#WEIGHT NETWORK
	weightNN = Input(shape = (15,29), name = "WeightSubNetworkInput")
	norminput_w = BatchNormalization()(weightNN)
	hidden_w1 = Dense(100, activation = 'relu')(norminput_w)
	norm_w1 = BatchNormalization()(hidden_w1)
	hidden_w2 = Dense(100, activation = 'relu')(norm_w1)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(hidden_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()	
	
   	 #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	hidden_m1 = Dense(300, activation = 'relu')(flat_via_reshape)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel
	
	
def createBNShHydrSubnetworks(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,29), name = "ResidueSubnetworkInput")
	norminput_p = BatchNormalization()(residueProbNN)
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,29))(norminput_p)
	norm_r1 = BatchNormalization()(hidden_r1)
	hidden_r2 = Dense(100, activation = 'relu')(norm_r1)
	norm_r2 = BatchNormalization()(hidden_r2)
	hidden_r3 = Dense(100, activation = 'relu')(norm_r2)
	norm_r3 = BatchNormalization()(hidden_r3)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(norm_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()
	
	#WEIGHT NETWORK
	weightNN = Input(shape = (15,29), name = "WeightSubNetworkInput")
	norminput_w = BatchNormalization()(weightNN)
	hidden_w1 = Dense(100, activation = 'relu')(norminput_w)
	norm_w1 = BatchNormalization()(hidden_w1)
	hidden_w2 = Dense(100, activation = 'relu')(norm_w1)
	norm_w2 = BatchNormalization()(hidden_w2)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(norm_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
	
   	 #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	hidden_m1 = Dense(300, activation = 'relu')(flat_via_reshape)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel
	

def shellAndHydrNN(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,29), name = "ResidueSubnetworkInput")
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,29))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu')(hidden_r1)
	hidden_r3 = Dense(100, activation = 'relu')(hidden_r2)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,29), name = "WeightSubNetworkInput")
	hidden_w1 = Dense(100, activation = 'relu')(weightNN)
	hidden_w2 = Dense(100, activation = 'relu')(hidden_w1)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(hidden_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
			
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	hidden_m1 = Dense(300, activation = 'relu')(flat_via_reshape)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel
	
	
def create_BSF_NN_skip_bn_ShandHydr(lr, m, nest, showsumm, showimage, lmda, dp):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,29))
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,29), kernel_regularizer = regularizers.l2(lmda))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_r1)
	combiner1r2 = concatenate([hidden_r1, hidden_r2])
	hidden_r3 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(combiner1r2)
	combiner2r3 = concatenate([combiner1r2, hidden_r3])
	batch_normed_r3 = BatchNormalization()(combiner2r3)
	dropped_r3 = Dropout(dp)(batch_normed_r3)
	resProbNNOut = Dense(20, activation = 'softmax', name = 'residueP', kernel_regularizer = regularizers.l2(lmda))(dropped_r3)
	print ("Size res out ", resProbNNOut.shape)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,29))
	hidden_w1 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(weightNN)
	hidden_w2 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_w1)
	combinew1w2 = concatenate([hidden_w1, hidden_w2])
	batch_normed_w2 = BatchNormalization()(combinew1w2)
	dropped_w2 = Dropout(dp)(batch_normed_w2)
	weightNNOut = Dense(1, activation = 'relu', name = 'weightN', kernel_regularizer = regularizers.l2(lmda))(dropped_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
			
    #OVERALL NETWORK
	merge_out = Multiply(name = 'merge_out')([weightNNOut, resProbNNOut])
	skiplayer = concatenate([merge_out, residueProbNN]) #15 * 47
	flat_via_reshape = Flatten(name = 'Concatenate')(skiplayer)
	batch_norm_skip = BatchNormalization()(flat_via_reshape)
	dropped_skip = Dropout(dp)(batch_norm_skip)
	#flat_via_reshape = Flatten(name = 'concat')(merge_out)
	hidden_m1 = Dense(705, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(dropped_skip)
	hidden_m2 = Dense(300, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_m1)
	dropped_m2 = Dropout(dp)(hidden_m2)
	hidden_m3 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(dropped_m2)
	combinem2m3 = concatenate([dropped_m2, hidden_m3])
	batch_normed_m3 = BatchNormalization()(combinem2m3)
	dropped_m3 = Dropout(dp)(batch_normed_m3)
	mergeOut = Dense(20, activation ='softmax', name = 'output_layer')(dropped_m3)
	sgd = SGD(lr = lr, momentum = m)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "bestRegularizationModelShAndHydr.svg")
	return mergeModel
	
	
#####################################
	
################# 3 shells on target, 3 shells and hydr on each neighbor shells at 20.0 30.0 and 40.0 angstroms
def createBN3ShHydrAll(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,34), name = "ResidueSubnetworkInput")
	norminput_p = BatchNormalization()(residueProbNN)
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,34))(norminput_p)
	norm_r1 = BatchNormalization()(hidden_r1)
	hidden_r2 = Dense(100, activation = 'relu')(norm_r1)
	
	norm_r2 = BatchNormalization()(hidden_r2)
	hidden_r3 = Dense(100, activation = 'relu')(norm_r2)
	norm_r3 = BatchNormalization()(hidden_r3)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(norm_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()
	
	#WEIGHT NETWORK
	weightNN = Input(shape = (15,34), name = "WeightSubNetworkInput")
	norminput_w = BatchNormalization()(weightNN)
	hidden_w1 = Dense(100, activation = 'relu')(norminput_w)
	norm_w1 = BatchNormalization()(hidden_w1)
	hidden_w2 = Dense(100, activation = 'relu')(norm_w1)
	
	norm_w2 = BatchNormalization()(hidden_w2)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(norm_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
	
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	norm_reshape = BatchNormalization()(flat_via_reshape)
	hidden_m1 = Dense(300, activation = 'relu')(norm_reshape)
	norm_m1 = BatchNormalization()(hidden_m1)
	hidden_m2 = Dense(100, activation = 'relu')(norm_m1)
	norm_m2 = BatchNormalization()(hidden_m2)
	hidden_m3 = Dense(100, activation = 'relu')(norm_m2)
	norm_m3 = BatchNormalization()(hidden_m3)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(norm_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel
	
	
def createBN3ShHydrAllDropout(lr, m, nest, showsumm, showimage, dp):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,34), name = "ResidueSubnetworkInput")
	norminput_p = BatchNormalization()(residueProbNN)
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,34))(norminput_p)
	norm_r1 = BatchNormalization()(hidden_r1)
	drop_r1 = Dropout(dp)(norm_r1)
	hidden_r2 = Dense(100, activation = 'relu')(drop_r1)
	norm_r2 = BatchNormalization()(hidden_r2)
	drop_r2 = Dropout(dp)(norm_r2)
	hidden_r3 = Dense(100, activation = 'relu')(drop_r2)
	norm_r3 = BatchNormalization()(hidden_r3)
	drop_r3 = Dropout(dp)(norm_r3)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(norm_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()
	
	#WEIGHT NETWORK
	weightNN = Input(shape = (15,34), name = "WeightSubNetworkInput")
	norminput_w = BatchNormalization()(weightNN)
	hidden_w1 = Dense(100, activation = 'relu')(norminput_w)
	norm_w1 = BatchNormalization()(hidden_w1)
	drop_w1 = Dropout(dp)(norm_w1)
	hidden_w2 = Dense(100, activation = 'relu')(drop_w1)
	norm_w2 = BatchNormalization()(hidden_w2)
	drop_w2 = Dropout(dp)(norm_w2)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(drop_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
	
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	norm_reshape = BatchNormalization()(flat_via_reshape)
	hidden_m1 = Dense(300, activation = 'relu')(norm_reshape)
	norm_m1 = BatchNormalization()(hidden_m1)
	drop_m1 = Dropout(dp)(norm_m1)
	hidden_m2 = Dense(100, activation = 'relu')(drop_m1)
	norm_m2 = BatchNormalization()(hidden_m2)
	drop_m2 = Dropout(dp)(norm_m2)
	hidden_m3 = Dense(100, activation = 'relu')(drop_m2)
	norm_m3 = BatchNormalization()(hidden_m3)
	drop_m3 = Dropout(dp)(norm_m3)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(drop_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel
	
	
def createBN3ShHydrInput(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,34), name = "ResidueSubnetworkInput")
	norminput_p = BatchNormalization()(residueProbNN)
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,34))(norminput_p)
	norm_r1 = BatchNormalization()(hidden_r1)
	hidden_r2 = Dense(100, activation = 'relu')(norm_r1)
	hidden_r3 = Dense(100, activation = 'relu')(hidden_r2)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()
	
	#WEIGHT NETWORK
	weightNN = Input(shape = (15,34), name = "WeightSubNetworkInput")
	norminput_w = BatchNormalization()(weightNN)
	hidden_w1 = Dense(100, activation = 'relu')(norminput_w)
	norm_w1 = BatchNormalization()(hidden_w1)
	hidden_w2 = Dense(100, activation = 'relu')(norm_w1)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(hidden_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()	
	
		
   	 #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	hidden_m1 = Dense(300, activation = 'relu')(flat_via_reshape)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel
	
	
def createBN3ShHydrSubnetworks(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,34), name = "ResidueSubnetworkInput")
	norminput_p = BatchNormalization()(residueProbNN)
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,34))(norminput_p)
	norm_r1 = BatchNormalization()(hidden_r1)
	hidden_r2 = Dense(100, activation = 'relu')(norm_r1)
	norm_r2 = BatchNormalization()(hidden_r2)
	hidden_r3 = Dense(100, activation = 'relu')(norm_r2)
	norm_r3 = BatchNormalization()(hidden_r3)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(norm_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()
	
	#WEIGHT NETWORK
	weightNN = Input(shape = (15,34), name = "WeightSubNetworkInput")
	norminput_w = BatchNormalization()(weightNN)
	hidden_w1 = Dense(100, activation = 'relu')(norminput_w)
	norm_w1 = BatchNormalization()(hidden_w1)
	hidden_w2 = Dense(100, activation = 'relu')(norm_w1)
	norm_w2 = BatchNormalization()(hidden_w2)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(norm_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
	
   	 #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	hidden_m1 = Dense(300, activation = 'relu')(flat_via_reshape)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel
	

def createNN3ShellAndHydr(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,34), name = "ResidueSubnetworkInput")
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,34))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu')(hidden_r1)
	hidden_r3 = Dense(100, activation = 'relu')(hidden_r2)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,34), name = "WeightSubNetworkInput")
	hidden_w1 = Dense(100, activation = 'relu')(weightNN)
	hidden_w2 = Dense(100, activation = 'relu')(hidden_w1)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(hidden_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
			
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	hidden_m1 = Dense(300, activation = 'relu')(flat_via_reshape)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardModel.png")
	return mergeModel
	
	
################################## Skip and psuedoResidual Networks #########################################	

def create_BSF_NN_skip_bn(lr, m, nest, showsumm, showimage, lmda, dp):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,27))
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,27), kernel_regularizer = regularizers.l2(lmda))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_r1)
	combiner1r2 = concatenate([hidden_r1, hidden_r2])
	hidden_r3 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(combiner1r2)
	combiner2r3 = concatenate([combiner1r2, hidden_r3])
	batch_normed_r3 = BatchNormalization()(combiner2r3)
	dropped_r3 = Dropout(dp)(batch_normed_r3)
	resProbNNOut = Dense(20, activation = 'softmax', name = 'residueP', kernel_regularizer = regularizers.l2(lmda))(dropped_r3)
	print ("Size res out ", resProbNNOut.shape)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,27))
	hidden_w1 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(weightNN)
	hidden_w2 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_w1)
	combinew1w2 = concatenate([hidden_w1, hidden_w2])
	batch_normed_w2 = BatchNormalization()(combinew1w2)
	dropped_w2 = Dropout(dp)(batch_normed_w2)
	weightNNOut = Dense(1, activation = 'relu', name = 'weightN', kernel_regularizer = regularizers.l2(lmda))(dropped_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
			
    #OVERALL NETWORK
	merge_out = Multiply(name = 'merge_out')([weightNNOut, resProbNNOut])
	skiplayer = concatenate([merge_out, residueProbNN]) #15 * 47
	flat_via_reshape = Flatten(name = 'Concatenate')(skiplayer)
	batch_norm_skip = BatchNormalization()(flat_via_reshape)
	dropped_skip = Dropout(dp)(batch_norm_skip)
	#flat_via_reshape = Flatten(name = 'concat')(merge_out)
	hidden_m1 = Dense(705, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(dropped_skip)
	hidden_m2 = Dense(500, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_m1)
	dropped_m2 = Dropout(dp)(hidden_m2)
	hidden_m3 = Dense(300, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(dropped_m2)
	combinem2m3 = concatenate([dropped_m2, hidden_m3])
	batch_normed_m3 = BatchNormalization()(combinem2m3)
	dropped_m3 = Dropout(dp)(batch_normed_m3)
	mergeOut = Dense(20, activation ='softmax', name = 'output_layer')(dropped_m3)
	sgd = SGD(lr = lr, momentum = m)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "bestRegularizationModel.svg")
	return mergeModel
	


def createNNPsuedoResidual(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,27), name = "ResidueSubnetworkInput")
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,27))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu')(hidden_r1)
	combiner1r2 = concatenate([hidden_r1, hidden_r2])
	hidden_r3 = Dense(100, activation = 'relu')(combiner1r2)
	combiner2r3 = concatenate([combiner1r2, hidden_r3])
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,27), name = "WeightSubNetworkInput")
	hidden_w1 = Dense(100, activation = 'relu')(weightNN)
	hidden_w2 = Dense(100, activation = 'relu')(hidden_w1)
	combinew1w2 = concatenate([hidden_w1, hidden_w2])
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(combinew1w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
			
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	hidden_m1 = Dense(300, activation = 'relu')(flat_via_reshape)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	combinem2m3 = concatenate([hidden_m2, hidden_m3])
	mergeOut = Dense(20, activation ='softmax', name = "Output")(combinem2m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "psuedoResidualNetwork.png")
	return mergeModel


def createNNPsuedoResidualWithSkip(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,27), name = "ResidueSubnetworkInput")
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,27))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu')(hidden_r1)
	combiner1r2 = concatenate([hidden_r1, hidden_r2])
	hidden_r3 = Dense(100, activation = 'relu')(combiner1r2)
	combiner2r3 = concatenate([combiner1r2, hidden_r3])
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,27), name = "WeightSubNetworkInput")
	hidden_w1 = Dense(100, activation = 'relu')(weightNN)
	hidden_w2 = Dense(100, activation = 'relu')(hidden_w1)
	combinew1w2 = concatenate([hidden_w1, hidden_w2])
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(combinew1w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
			
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	skiplayer = concatenate([merge_out, residueProbNN]) #15 * 47
	flat_via_reshape = Flatten(name = 'Concatenate')(skiplayer)
	#flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	hidden_m1 = Dense(300, activation = 'relu')(flat_via_reshape)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	combinem2m3 = concatenate([hidden_m2, hidden_m3])
	mergeOut = Dense(20, activation ='softmax', name = "Output")(combinem2m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "psuedoResidualNetwork.png")
	return mergeModel


###################### Long Input Networks (using 3 long one hot with secondary structure instead of 1,2,3) ###############################3

def createNNLongInput(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,31), name = "ResidueSubnetworkInput")
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,31))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu')(hidden_r1)
	hidden_r3 = Dense(100, activation = 'relu')(hidden_r2)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,31), name = "WeightSubNetworkInput")
	hidden_w1 = Dense(100, activation = 'relu')(weightNN)
	hidden_w2 = Dense(100, activation = 'relu')(hidden_w1)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(hidden_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
			
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'Concatenate')(merge_out)
	hidden_m1 = Dense(300, activation = 'relu')(flat_via_reshape)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "LongInputSizeStandardModel.png")
	return mergeModel
	
	
###################################### Dense Feedforward Network #######################################################

def createDenseNN(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (293,))
	hidden_m1 = Dense(293, activation = 'relu', input_shape = (293,))(residueProbNN)
	hidden_m2 = Dense(293, activation = 'relu')(hidden_m1)
	mergeOut = Dense(20, activation ='softmax', name = 'output_layer')(hidden_m2)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	mergeModel = Model(inputs = residueProbNN, outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		try:
			plot_model(mergeModel, show_shapes = True, to_file = "simpleDenseNetwork.png")
		except:
			print ("WILL NOT PLOT- pydot errors on this computer")
	return mergeModel


################################ Window Subnetwork Networks ###################################################################3
def createWindowNN(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,27), name = "ResidueSubnetworkInput")
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,27))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu')(hidden_r1)
	hidden_r3 = Dense(100, activation = 'relu')(hidden_r2)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,27), name = "WeightSubNetworkInput")
	hidden_w1 = Dense(100, activation = 'relu')(weightNN)
	hidden_w2 = Dense(100, activation = 'relu')(hidden_w1)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(hidden_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()

	#WINDOW NETWORK
	#Based on this: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5009531/
	windowNN = Input(shape= (282,), name = "WindowNetworkInput")
	hidden_win1 = Dense(500, activation = 'relu', input_shape = (282,))(windowNN)
	windowOut = Dense(20, activation = 'softmax')(hidden_win1)
	windowNNMod = Model(windowNN, windowOut)
	windowNNMod.summary()
	
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'FlattenMultiplications')(merge_out)
	concat = concatenate([flat_via_reshape, windowOut])
	hidden_m1 = Dense(320, activation = 'relu')(concat)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN, windowNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardNetworkWithWindow.png")
	return mergeModel
	
	
def createWindowSmallNN(lr, m, nest, showsumm, showimage):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,27), name = "ResidueSubnetworkInput")
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,27))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu')(hidden_r1)
	hidden_r3 = Dense(100, activation = 'relu')(hidden_r2)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,27), name = "WeightSubNetworkInput")
	hidden_w1 = Dense(100, activation = 'relu')(weightNN)
	hidden_w2 = Dense(100, activation = 'relu')(hidden_w1)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(hidden_w2)
	weightNNMOd = Model(weightNN, weightNNcombinedshort90Pertrain4000val1000NonNormedWeightsNestFalseCvcv1Out)
	weightNNMOd.summary()

	#WINDOW NETWORK
	#Based on this: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5009531/
	windowNN = Input(shape= (282,), name = "WindowNetworkInput")
	hidden_win1 = Dense(282, activation = 'relu', input_shape = (282,))(windowNN)
	windowOut = Dense(20, activation = 'softmax')(hidden_win1)
	windowNNMod = Model(windowNN, windowOut)
	windowNNMod.summary()
	
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'FlattenMultiplications')(merge_out)
	concat = concatenate([flat_via_reshape, windowOut])
	hidden_m1 = Dense(320, activation = 'relu')(concat)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN, windowNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardNetworkWithWindow.png")
	return mergeModel
	
	
def createWindowSmallDropNN(lr, m, nest, showsumm, showimage, dp):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,27), name = "ResidueSubnetworkInput")
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,27))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu')(hidden_r1)
	hidden_r3 = Dense(100, activation = 'relu')(hidden_r2)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,27), name = "WeightSubNetworkInput")
	hidden_w1 = Dense(100, activation = 'relu')(weightNN)
	hidden_w2 = Dense(100, activation = 'relu')(hidden_w1)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(hidden_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()

	#WINDOW NETWORK
	#Based on this: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5009531/
	windowNN = Input(shape= (282,), name = "WindowNetworkInput")
	hidden_win1 = Dense(282, activation = 'relu', input_shape = (282,))(windowNN)
	batchnorm_win1 = BatchNormalization()(hidden_win1)
	dropped_win1 = Dropout(dp)(batchnorm_win1)
	windowOut = Dense(20, activation = 'softmax')(dropped_win1)
	windowNNMod = Model(windowNN, windowOut)
	windowNNMod.summary()
		
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'FlattenMultiplications')(merge_out)
	concat = concatenate([flat_via_reshape, windowOut])
	hidden_m1 = Dense(320, activation = 'relu')(concat)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN, windowNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardNetworkWithWindow.png")
	return mergeModel
	
	
def createWindowNN_batchNormDropout(lr, m, nest, showsumm, showimage, dp):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,27), name = "ResidueSubnetworkInput")
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,27))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu')(hidden_r1)
	hidden_r3 = Dense(100, activation = 'relu')(hidden_r2)
	resProbNNOut = Dense(20, activation = 'softmax', name = "ResidueSubnetworkOutput")(hidden_r3)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,27), name = "WeightSubNetworkInput")
	hidden_w1 = Dense(100, activation = 'relu')(weightNN)
	hidden_w2 = Dense(100, activation = 'relu')(hidden_w1)
	weightNNOut = Dense(1, activation = 'relu', name = "WeightSubNetworkOutput")(hidden_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()

	#WINDOW NETWORK
	#Based on this: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5009531/
	windowNN = Input(shape= (282,), name = "WindowNetworkInput")
	hidden_win1 = Dense(500, activation = 'relu', input_shape = (282,))(windowNN)
	batch_norm_win1 = BatchNormalization()(hidden_win1)
	dropped_win1 = Dropout(dp)(batch_norm_win1)
	windowOut = Dense(20, activation = 'softmax')(dropped_win1)
	windowNNMod = Model(windowNN, windowOut)
	windowNNMod.summary()
	
    #OVERALL NETWORK
	merge_out = Multiply(name = 'CrossMultiplication')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'FlattenMultiplications')(merge_out)
	concat = concatenate([flat_via_reshape, windowOut])
	hidden_m1 = Dense(320, activation = 'relu')(concat)
	hidden_m2 = Dense(100, activation = 'relu')(hidden_m1)
	hidden_m3 = Dense(100, activation = 'relu')(hidden_m2)
	mergeOut = Dense(20, activation ='softmax', name = "Output")(hidden_m3)
	sgd = SGD(lr = lr, momentum = m, nesterov = nest)
	#adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	mergeModel = Model(inputs = [residueProbNN,weightNN, windowNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "standardNetworkWithWindow.png")
	return mergeModel


def create_BSF_NN(lr, m, nest, showsumm, showimage, lmda, dp):
	# RESIDUE NETWORK
	residueProbNN = Input(shape = (15,27))
	hidden_r1 = Dense(100, activation = 'relu', input_shape = (15,27), kernel_regularizer = regularizers.l2(lmda))(residueProbNN)
	hidden_r2 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_r1)
	hidden_r3 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_r2)
	batch_normed_r3 = BatchNormalization()(hidden_r3)
	dropped_r3 = Dropout(dp)(batch_normed_r3)
	resProbNNOut = Dense(20, activation = 'softmax', name = 'residueP', kernel_regularizer = regularizers.l2(lmda))(dropped_r3)
	print ("Size res out ", resProbNNOut.shape)
	residueNNMod = Model(residueProbNN,resProbNNOut)
	residueNNMod.summary()

	#WEIGHT NETWORK
	weightNN = Input(shape = (15,27))
	hidden_w1 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(weightNN)
	hidden_w2 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_w1)
	batch_normed_w2 = BatchNormalization()(hidden_w2)
	dropped_w2 = Dropout(dp)(batch_normed_w2)
	weightNNOut = Dense(1, activation = 'relu', name = 'weightN', kernel_regularizer = regularizers.l2(lmda))(dropped_w2)
	weightNNMOd = Model(weightNN, weightNNOut)
	weightNNMOd.summary()
			
    #OVERALL NETWORK
	merge_out = Multiply(name = 'merge_out')([weightNNOut, resProbNNOut])
	flat_via_reshape = Flatten(name = 'concat')(merge_out)
	hidden_m1 = Dense(300, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(flat_via_reshape)
	hidden_m2 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(hidden_m1)
	dropped_m2 = Dropout(dp)(hidden_m2)
	hidden_m3 = Dense(100, activation = 'relu', kernel_regularizer = regularizers.l2(lmda))(dropped_m2)
	batch_normed_m3 = BatchNormalization()(hidden_m3)
	dropped_m3 = Dropout(dp)(batch_normed_m3)
	mergeOut = Dense(20, activation ='softmax', name = 'output_layer')(dropped_m3)
	sgd = SGD(lr = lr, momentum = m)
	mergeModel = Model(inputs = [residueProbNN,weightNN], outputs = mergeOut)
	mergeModel.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	if showsumm:
		mergeModel.summary()
	if showimage:
		print ("WILL NOT PLOT- pydot errors on this computer")
		plot_model(mergeModel, show_shapes = True, to_file = "bestRegularizationModel.png")
	return mergeModel
	
	
######################Training Functions ##########################################################
#Function to train with fit generator method
def training(mModel, training_files, validation_files, fit_batch_size, validate_batch_size, weight_set, long_or_short, binary_files, mType, save_data_every_x_epochs, basename, continuous_training, graph_acc, graph_loss, sequential, end_sims, dataset, even_uneven, original_models,window_models, simple_dense_models, sh3andHydr_models, shandHydr_models, number_final_validations_for_end_sims):
	storage = {'acc':[], 'val_acc':[], 'loss':[], 'val_loss':[]}
	i = 1
	#Declaring fit generator and validation generator depending on the model type
	if sequential and mType == "m": 
		print ("Getting SEQUENTIAL	generators")
		fit_gen = generate_arrays_from_protein_Files(training_files, fit_batch_size, weight_set, long_or_short, binary_files)
		valid_gen = generate_arrays_from_protein_Files(validation_files, validate_batch_size, weight_set, long_or_short, binary_files)
	elif mType in window_models:
		print ("Getting WINDOW TYPE generators")
		fit_gen = generate_arrays_for_window_from_protein_files(training_files, fit_batch_size, weight_set, long_or_short, binary_files, even_uneven)
		valid_gen = generate_arrays_for_window_from_protein_files(validation_files, validate_batch_size, weight_set, long_or_short, binary_files, even_uneven)
	elif mType in original_models: 
		print ("Getting TEXT FILE generators")
		fit_gen = generate_arrays_from_text_files(training_files, fit_batch_size, weight_set, dataset, long_or_short)
		valid_gen = generate_arrays_from_text_files(validation_files, validate_batch_size, weight_set, dataset, long_or_short)
	elif mType in shandHydr_models:
		print ("Getting 1 sphere and neighbor hydroph generators")
		fit_gen =  generate_arrays_from_Protein_Files_shHydr(training_files, fit_batch_size, weight_set, long_or_short, binary_files)
		valid_gen =  generate_arrays_from_Protein_Files_shHydr(validation_files, validate_batch_size, weight_set, long_or_short, binary_files)
	elif mType in sh3andHydr_models:
		if sequential:
			print ("Getting 3 sphere and neighbor hydroph generators")
			fit_gen = generate_arrays_from_Protein_Files_3shHydr(training_files, fit_batch_size, weight_set, long_or_short, binary_files)
			valid_gen = generate_arrays_from_Protein_Files_3shHydr(validation_files, validate_batch_size, weight_set, long_or_short, binary_files)
		else:
			print ("Getting 3 sphere and neighbor hydrophobicity text file generators")
			fit_gen = generate_arrays_from_text_files_3shHydr(training_files, fit_batch_size, weight_set, long_or_short, binary_files)
			valid_gen = generate_arrays_from_text_files_3shHydr(validation_files, validate_batch_size, weight_set, long_or_short, binary_files)
	elif mType in simple_dense_models:
		print ("Getting simple dense generators")
		fit_gen = generate_long_arrays_from_text_files(training_files, fit_batch_size, weight_set, dataset, long_or_short)
		valid_gen = generate_long_arrays_from_text_files(validation_files, validate_batch_size, weight_set, dataset, long_or_short)
	else:
		print ("WARNING!  HALTING TRAINING!  NO GENERATOR TYPE WAS ASSIGNED TO THIS MODEL (if new, will need to be added to correct generator list in paramter function")
		pause = input("Waiting for you to close....")
	mv = 1
	#Trains until manually stopped
	if continuous_training: 
		while 1:
			print ("Model: ", basename, " Epoch: ", i)
			history = mModel.fit_generator(generator = fit_gen, steps_per_epoch = 1, epochs = 1, verbose = mv, validation_data = valid_gen, validation_steps = 1, max_queue_size=8, workers=1,use_multiprocessing=False)
			storage['acc'] = storage['acc'] + history.history['acc']
			storage['val_acc'] = storage['val_acc'] + history.history['val_acc']
			storage['loss'] = storage['loss'] + history.history['loss']
			storage['val_loss'] = storage['val_loss'] + history.history['val_loss']
			#escape and enter to exit continual training
			if kbhit() :
				if getch() == chr(27) :
					print("Got escape keypress")
					break
			if i%save_data_every_x_epochs == 0:
				store_acc = open(basename + "ValidationLog.txt", "a")
				store_loss = open(basename + "LossLog.txt", "a")
				#Autosave and generate training graph
				acc_to_add = storage['acc'][-save_data_every_x_epochs:]
				val_acc_to_add = storage['val_acc'][-save_data_every_x_epochs:]
				loss_to_add = storage['loss'][-save_data_every_x_epochs:]
				val_loss_to_add = storage['val_loss'][-save_data_every_x_epochs:]
				for k in range(0, len(acc_to_add)):
					store_acc.write(str(acc_to_add[k]) + "," + str(val_acc_to_add[k]) + "\n")
					store_loss.write(str(loss_to_add[k]) + "," + str(val_loss_to_add[k]) + "\n")
				store_acc.close()
				store_loss.close()
				makePlots(basename, graph_acc, graph_loss, storage, i)
			i += 1
		store_acc = open(basename + "ValidationLog.txt", "a")
		store_loss = open(basename + "LossLog.txt", "a")
		#Autosave and generate training graph
		if not len(storage['acc'])%save_data_every_x_epochs == 0:
			to_get = len(storage['acc']) - (len(storage['acc'])//save_data_every_x_epochs)*save_data_every_x_epochs
		else:
			to_get = -1
		if not to_get == -1:
			acc_to_add = storage['acc'][-to_get:]
			val_acc_to_add = storage['val_acc'][-to_get:]
			loss_to_add = storage['loss'][-to_get:]
			val_loss_to_add = storage['val_loss'][-to_get:]
			for k in range(0, len(acc_to_add)):
				#lines +=1
				#print ("lines: ", lines)
				store_acc.write(str(acc_to_add[k]) + "," + str(val_acc_to_add[k]) + "\n")
				store_loss.write(str(loss_to_add[k]) + "," + str(val_loss_to_add[k]) + "\n")
		store_acc.close()
		store_loss.close()
		makePlotsLoadAll(basename)
	else:
	#training a set number of epochs (end_sims epochs total)
		for j in range(0,end_sims):
			print ("Model: ", basename, " Epoch: ", j + 1, " out of " + str(end_sims) )
			history = mModel.fit_generator(generator = fit_gen, steps_per_epoch = 1, epochs = 1, verbose = mv, validation_data = valid_gen, validation_steps = 1, max_queue_size=8, workers=1,use_multiprocessing=False)
			#store data
			storage['acc'] = storage['acc'] + history.history['acc']
			storage['val_acc'] = storage['val_acc'] + history.history['val_acc']
			storage['loss'] = storage['loss'] + history.history['loss']
			storage['val_loss'] = storage['val_loss'] + history.history['val_loss']
			if kbhit() :
				if getch() == chr(27) :
					print("Got escape keypress")
					break
			if j%save_data_every_x_epochs == 0:
				store_acc = open(basename + "ValidationLog.txt", "a")
				store_loss = open(basename + "LossLog.txt", "a")
				#Autosave and generate training graph
				acc_to_add = storage['acc'][-save_data_every_x_epochs:]
				val_acc_to_add = storage['val_acc'][-save_data_every_x_epochs:]
				loss_to_add = storage['loss'][-save_data_every_x_epochs:]
				val_loss_to_add = storage['val_loss'][-save_data_every_x_epochs:]
				for k in range(0, len(acc_to_add)):
					store_acc.write(str(acc_to_add[k]) + "," + str(val_acc_to_add[k]) + "\n")
					store_loss.write(str(loss_to_add[k]) + "," + str(val_loss_to_add[k]) + "\n")
				store_acc.close()
				store_loss.close()
				makePlots(basename, graph_acc, graph_loss, storage, j)
				#Autosave and generate training graph
		#saving remaining data since last save
		store_acc = open(basename + "ValidationLog.txt", "a")
		store_loss = open(basename + "LossLog.txt", "a")
		if not len(storage['acc'])%save_data_every_x_epochs == 0:
			to_get = len(storage['acc']) - (len(storage['acc'])//save_data_every_x_epochs)*save_data_every_x_epochs
		else:
			to_get = -1
		if not to_get == -1:
			acc_to_add = storage['acc'][-to_get:]
			val_acc_to_add = storage['val_acc'][-to_get:]
			loss_to_add = storage['loss'][-to_get:]
			val_loss_to_add = storage['val_loss'][-to_get:]
			for k in range(0, len(acc_to_add)):
				#lines +=1
				#print ("lines: ", lines)
				store_acc.write(str(acc_to_add[k]) + "," + str(val_acc_to_add[k]) + "\n")
				store_loss.write(str(loss_to_add[k]) + "," + str(val_loss_to_add[k]) + "\n")
		store_acc.close()
		store_loss.close()
		#generate final graphs
		makePlotsLoadAll(basename)
		#get val acc and loss for entire validation dataset for cross validation 
		final_acc = mModel.evaluate_generator(valid_gen, number_final_validations_for_end_sims)
		final_acc_file = open(basename + "finalAccAndLoss.txt", "w")
		final_acc_file.write(mModel.metrics_names[0] +" " + str(final_acc[0]) + "\n")
		final_acc_file.write(mModel.metrics_names[1] +" "+ str(final_acc[1]) + "\n")
		final_acc_file.close()


#trains model with fit_batch method (fits on a batch from the generator at a time)
#slightly faster than fit_generator

def batchTraining(mModel, training_files, validation_files, fit_batch_size, validate_batch_size, weight_set, long_or_short, binary_files, mType, save_data_every_x_epochs, basename, continuous_training, graph_acc, graph_loss, sequential, end_sims, dataset, even_uneven, original_models,window_models, simple_dense_models, sh3andHydr_models, shandHydr_models, number_final_validations_for_end_sims):
	storage = {'acc':[], 'val_acc':[], 'loss':[], 'val_loss':[]}
	i = 1
	if not sequential:
		fit_gen = generate_arrays_from_text_files(training_files, fit_batch_size, weight_set, dataset, long_or_short)
		valid_gen = generate_arrays_from_text_files(validation_files, validate_batch_size, weight_set, dataset, long_or_short)
	else:
		if mType in sh3andHydr_models:
			print ("Getting 3 sphere and neighbor hydroph generators")
			fit_gen = generate_arrays_from_Protein_Files_3shHydr(training_files, fit_batch_size, weight_set, long_or_short, binary_files)
			valid_gen = generate_arrays_from_Protein_Files_3shHydr(validation_files, validate_batch_size, weight_set, long_or_short, binary_files)
		elif mType in shandHydr_models:
			print ("Getting 1 sphere and neighbor hydroph generators")
			fit_gen =  generate_arrays_from_Protein_Files_shHydr(training_files, fit_batch_size, weight_set, long_or_short, binary_files)
			valid_gen =  generate_arrays_from_Protein_Files_shHydr(validation_files, validate_batch_size, weight_set, long_or_short, binary_files)
		else:
			print ("getting sequential generators")
			fit_gen = generate_arrays_from_Protein_Files(training_files, fit_batch_size, weight_set, long_or_short, binary_files)
			valid_gen = generate_arrays_from_Protein_Files(validation_files, validate_batch_size, weight_set, long_or_short, binary_files)
	if continuous_training:	
		while 1:
			mv = 1
			print("EPOCH: ", i)
			all_train = next(fit_gen)
			all_valid = next(valid_gen)
			#x's
			x = all_train[0]
			x_valid = all_valid[0]
			#y's
			y = all_train[1]
			y_valid = all_valid[1]
			if not weight_set == 0:		
				#w's
				w = all_train[2]
				w_valid = all_valid[2]
			#train and validate
			if not weight_set == 0:
				hist_list = mModel.train_on_batch(x,y,sample_weight = w)
				valid_hist_list = mModel.test_on_batch(x_valid, y_valid)#, sample_weight = w_valid)
			else:
				hist_list = mModel.train_on_batch(x,y)
				valid_hist_list = mModel.test_on_batch(x_valid, y_valid)#, sample_weight = w_valid)
			#display model and stats for the epoch
			print ("model: ", basename)
			print ("Loss: ", hist_list[0], " Acc: ", hist_list[1], " Val Loss: ", valid_hist_list[0], " Val Acc: ", valid_hist_list[1])
			#store epoch data
			storage['acc'].append(hist_list[1])
			storage['val_acc'].append(valid_hist_list[1])
			storage['loss'].append(hist_list[0])
			storage['val_loss'].append(valid_hist_list[1])
			if kbhit() :
				if getch() == chr(27) :
					print("Got escape keypress")
					break
			#save data every x epochs
			if i%save_data_every_x_epochs == 0:
				store_acc = open(basename + "ValidationLog.txt", "a")
				store_loss = open(basename + "LossLog.txt", "a")
				#Autosave and generate training graph
				acc_to_add = storage['acc'][-save_data_every_x_epochs:]
				val_acc_to_add = storage['val_acc'][-save_data_every_x_epochs:]
				loss_to_add = storage['loss'][-save_data_every_x_epochs:]
				val_loss_to_add = storage['val_loss'][-save_data_every_x_epochs:]
				for k in range(0, len(acc_to_add)):
					store_acc.write(str(acc_to_add[k]) + "," + str(val_acc_to_add[k]) + "\n")
					store_loss.write(str(loss_to_add[k]) + "," + str(val_loss_to_add[k]) + "\n")
				store_acc.close()
				store_loss.close()
				makePlots(basename, graph_acc, graph_loss, storage, i)
			i += 1
		#save data remaining after last save when epochs finished
		store_acc = open(basename + "ValidationLog.txt", "a")
		store_loss = open(basename + "LossLog.txt", "a")
		if not len(storage['acc'])%save_data_every_x_epochs == 0:
			to_get = len(storage['acc']) - (len(storage['acc'])//save_data_every_x_epochs)*save_data_every_x_epochs
		else:
			to_get = -1
		if not to_get == -1:
			acc_to_add = storage['acc'][-to_get:]
			val_acc_to_add = storage['val_acc'][-to_get:]
			loss_to_add = storage['loss'][-to_get:]
			val_loss_to_add = storage['val_loss'][-to_get:]
			for k in range(0, len(acc_to_add)):
				store_acc.write(str(acc_to_add[k]) + "," + str(val_acc_to_add[k]) + "\n")
				store_loss.write(str(loss_to_add[k]) + "," + str(val_loss_to_add[k]) + "\n")
		store_acc.close()
		store_loss.close()
		makePlotsLoadAll(basename)
	else:
		#noncontinuous fit batch training
		for i in range(0,end_sims):
			mv = 1
			print("EPOCH: ", i, " of ", str(end_sims))
			print ("model: ", basename)
			all_train = next(fit_gen)
			all_valid = next(valid_gen)
			#x's
			x = all_train[0]
			x_valid = all_valid[0]
			#y's
			y = all_train[1]
			y_valid = all_valid[1]
			if not weight_set == 0:		
				#w's
				w = all_train[2]
				w_valid = all_valid[2]
			#train and validate
			if not weight_set == 0:
				hist_list = mModel.train_on_batch(x,y,sample_weight = w)
				valid_hist_list = mModel.test_on_batch(x_valid, y_valid)#, sample_weight = w_valid)
			else:
				hist_list = mModel.train_on_batch(x,y)
				valid_hist_list = mModel.test_on_batch(x_valid, y_valid)#, sample_weight = w_valid)
			#display epoch data
			print ("Loss: ", hist_list[0], " Acc: ", hist_list[1], " Val Loss: ", valid_hist_list[0], " Val Acc: ", valid_hist_list[1])
			#store epoch data
			storage['acc'].append(hist_list[1])
			storage['val_acc'].append(valid_hist_list[1])
			storage['loss'].append(hist_list[0])
			storage['val_loss'].append(valid_hist_list[1])
			if kbhit() :
				if getch() == chr(27) :
					print("Got escape keypress")
					break
			#save data every x epochs
			if i%save_data_every_x_epochs == 0:
				store_acc = open(basename + "ValidationLog.txt", "a")
				store_loss = open(basename + "LossLog.txt", "a")
				#Autosave and generate training graph
				acc_to_add = storage['acc'][-save_data_every_x_epochs:]
				val_acc_to_add = storage['val_acc'][-save_data_every_x_epochs:]
				loss_to_add = storage['loss'][-save_data_every_x_epochs:]
				val_loss_to_add = storage['val_loss'][-save_data_every_x_epochs:]
				for k in range(0, len(acc_to_add)):
					store_acc.write(str(acc_to_add[k]) + "," + str(val_acc_to_add[k]) + "\n")
					store_loss.write(str(loss_to_add[k]) + "," + str(val_loss_to_add[k]) + "\n")
				store_acc.close()
				store_loss.close()
				makePlots(basename, graph_acc, graph_loss, storage, i)
			i += 1
		#store remaining data after end_sims epochs
		store_acc = open(basename + "ValidationLog.txt", "a")
		store_loss = open(basename + "LossLog.txt", "a")
		if not len(storage['acc'])%save_data_every_x_epochs == 0:
			to_get = len(storage['acc']) - (len(storage['acc'])//save_data_every_x_epochs)*save_data_every_x_epochs
		else:
			to_get = -1
		if not to_get == -1:
			acc_to_add = storage['acc'][-to_get:]
			val_acc_to_add = storage['val_acc'][-to_get:]
			loss_to_add = storage['loss'][-to_get:]
			val_loss_to_add = storage['val_loss'][-to_get:]
			for k in range(0, len(acc_to_add)):
				store_acc.write(str(acc_to_add[k]) + "," + str(val_acc_to_add[k]) + "\n")
				store_loss.write(str(loss_to_add[k]) + "," + str(val_loss_to_add[k]) + "\n")
		store_acc.close()
		store_loss.close()
		makePlotsLoadAll(basename)
		#create cross validation acc and loss for statistics in sep. text file
		final_acc = mModel.evaluate_generator(valid_gen, number_final_validations_for_end_sims)
		final_acc_file = open(basename + "finalAccAndLoss.txt", "w")
		final_acc_file.write(mModel.metrics_names[0] +" " + str(final_acc[0]) + "\n")
		final_acc_file.write(mModel.metrics_names[1] +" "+ str(final_acc[1]) + "\n")
		final_acc_file.close()

#####################

##########################################         Graphing Functions   #######################################

#Control function for intermediate graphing 
def makePlots(basename, graph_acc, graph_loss, storage, epochs):
	if graph_acc:
		makePlotsAccuracy(storage, basename, epochs)	
	if graph_loss:
		makePlotsLoss(storage, basename, epochs)


#graphing function at end of training, loads all past training from logs to graph all data
def makePlotsLoadAll(basename):
	#graphing acc data
	store_acc = open(basename + "ValidationLog.txt", "r")
	s_a = store_acc.read().splitlines()
	acc =[]
	val_acc = []
	for thing in s_a:
		both = thing.split(",")
		acc.append(float(both[0]) * 100)
		val_acc.append(float(both[1]) * 100)
		title = basename + "trainingAcc.png"
	plt.figure(3)
	a, = plt.plot(acc, label = "Training")
	va, = plt.plot(val_acc, label = "Validation")
	plt.title(basename + str(len(acc)) + " Epochs Final Accuracy")
	plt.ylabel('Accuracy (%)')
	plt.xlabel('Epoch')
	plt.legend(handles =[a,va], loc='upper left')
	plt.savefig(basename + "FinalAccGraph.png")
	#graphing loss
	store_loss = open(basename + "LossLog.txt", "r")
	s_l = store_loss.read().splitlines()
	loss =[]
	val_loss = []
	for thing in s_l:
		both = thing.split(",")
		loss.append(float(both[0]) * 100)
		val_loss.append(float(both[1]) * 100)
		title = basename + "trainingLoss.png"
	plt.figure(4)
	l, = plt.plot(loss)
	vl, = plt.plot(val_loss)
	plt.title(basename + str(len(loss)) +  " Epochs Final Loss")
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(handles = [l, vl], loc='upper left')
	plt.savefig(basename + "FinalLossGraph.png")

#Acc graphs
def makePlotsAccuracy(hist_dict_manual, basename, epochs):
	title = basename  +"TrainingAcc.png"
	plt.figure(1)
	t, = plt.plot(hist_dict_manual['acc'], label = "Training")
	v, = plt.plot(hist_dict_manual['val_acc'], label = "Validation")
	plt.title(basename + " Accuracy for " + str(epochs) + " Epochs" )
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(handles = [t, v], loc='upper left')
	plt.savefig(title)
	
#Loss and Validation Loss graphs
def makePlotsLoss(hist_dict_manual, basename, epochs):
	title = basename  + "TrainingLoss.png"
	plt.figure(2)
	train, = plt.plot(hist_dict_manual['loss'], label = "Training")
	val, = plt.plot(hist_dict_manual['val_loss'], label = "Validation")
	plt.title(basename + " Loss for " + str(epochs) + " Epochs")
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(handles = [train, val], loc='upper left')
	plt.savefig(title)

#############

############################################        Save model function ########################################
def save_model(mergeModel, yaml_name, h5_name):
	model_yaml = mergeModel.to_yaml()
	with open(yaml_name, "w") as yaml_file:
		yaml_file.write(model_yaml)
	yaml_file.close()
	# serialize weights to HDF5
	mergeModel.save_weights(h5_name)
	print("Saved ", yaml_name, " and ", h5_name)

################

#########################################################################################################
##
##												MAIN 
##
#########################################################################################################

set_parameters_and_run()









