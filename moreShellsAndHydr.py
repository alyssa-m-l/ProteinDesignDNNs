#AML 6/8/18
#Prepare test data set- final
#Imports and PyRosestta init
#!/usr/bin/python

from pyrosetta import *
init()

import matplotlib.pyplot as plt
from pyrosetta import toolbox
import rosetta.core.scoring.sasa as sasa
import rosetta.core.scoring.hbonds as hbonds
import numpy as numpy #for data output as a numpy array file, format is numpy.
from queue import PriorityQueue
from pyrosetta.rosetta.core.pose import compute_unique_chains
from math import sqrt, cos, sin
import numpy as np
import os 
import sys
from multiprocessing import Pool, current_process
dataPath = os.getcwd()



#using third column from table 1 from http://blanco.biomol.uci.edu/hydrophobicity_scales.html (octanol-interface scale)
#assuming ph=7.0 (hist is uncharged)
hydrophobicities = {'A':0.33 , 'R':1.00, 'N':0.43, 'D':2.41, 'C':0.22, 'Q':0.19, 'E':1.61, "G":1.14, "H":-0.06, "I":-0.81, "L":-0.69, "K":1.81, "M":-0.44, "F":-0.58, "P":-0.31, "S":0.33, "T":0.11 , "W": -0.24, "Y": 0.23, "V": -0.53}




#Creates N=15 clusters based on Calpha-Calpha distance from all residues in the protein 
#Input: pyrosetta pose obejct of the protein
#Output: Dictionary w/ key of target residue pose res_id # and values of 15 closest neighbor pose res_id #
#TODO: ONLY RUN THIS ONCE
def getAllDistanceMatrix(pose, cutoff1, cutoff2, cutoff3):
	residues = {} #will be a dictionary of residue and fifteen closest neighbor residues
	#Will use PriorityQueeu class to get lowest 15
	sphere_counts = {}
	for r in range(1,pose.total_residue() + 1):
		pri_q = PriorityQueue()
		sphere_counts[r] = [0,0,0]
		for rsub in range(1,pose.total_residue()+1):
			if not rsub == r:
				ca_r = pose.residue(r).xyz('CA') #
				ca_rsub = pose.residue(rsub).xyz('CA')
				dist = sqrt((ca_r.x - ca_rsub.x)**2 + (ca_r.y - ca_rsub.y)**2 + (ca_r.z - ca_rsub.z)**2)
				if dist < cutoff1:
					sphere_counts[r][0] +=1
				if dist < cutoff2:
					sphere_counts[r][1] += 1
				if dist < cutoff3:
					sphere_counts[r][2] += 1
				#print (dist, " ", rsub)
				pri_q.put((dist, rsub))
				
		#get 15 closest neighbors
		#cutoff distance is 30 Angstroms for sphere, getting neighbors in the sphere
		residues[r] = []
		for i in range (1,16):
			stored = pri_q.get()
			st = list(stored)
			#if st[1] < cutoff:
				#sphere_counts[r] +=1
			residues[r].append(st[1])
		'''
		#updates sphere if more residues fall into it
		continue_popping = True
		while continue_popping:
			x = pri_q.get()
			st = list(stored)
			if st[1] < cutoff:
				continue_popping = True
				sphere_counts[r] +=1
			else:
				continue_popping = False
		'''
	return residues, sphere_counts

#Creates a pyrosetta vector of all backbone SASA values for the residue (Ca, C, N, and O SASA values)
#Input: pose object of pro
#Output: a pyrosetta vector of the values
#TODO: ONLY RUN THIS ONCE		
def getAllbbSASA(pose):
	calc = sasa.SasaCalc()
	calc.calculate(pose)
	return calc.get_residue_sasa_bb() 


#get the number representation of Q3 secondary structure
#input: pose object of the protein, the pose res_id # for the residue you want the secondary structure for
#output: Whole number repres of the Q3 (1 =E, 2 = H, 3 = L)
def getQ3(pose, resid):
	char_ver = pose.conformation().secstruct(resid) #returns the secondary structure as L, H, or E 
	char_to_numb = {'E':1, 'H':2, 'L':3}
	return char_to_numb[char_ver] #returns the number representation

#returns all hbonds in a dictionary, residue pose number key and list of donors and acceptors contents per key
#input of pose object for protein
#output of dictionary
#needs hbonds imported as hbonds from rosetta.core.scoring to work
#TODO: ONLY RUN ONCE
def getAllHBonds(pose):
    pose_hbonds = hbonds.HBondSet(pose, False) #gets all bb-bb only H-bonds for the pos
    hbond_dictionary = {}
    total_r = pose.total_residue()
    total_rplus = total_r + 1
    for r in range(1,total_rplus):
        hbond_dictionary[r] = []
        for h in range(1,pose_hbonds.nhbonds()+1):
            hb = pose_hbonds.hbond(h)
            a_res = hb.acc_res() #acceptor
            d_res = hb.don_res() #donor
            if r == a_res:
                hbond_dictionary[r].append(d_res)
            if r == d_res:
                hbond_dictionary[r].append(a_res)
    #
    return hbond_dictionary

#gets all three unit vector xyz portions
#input: pose.residue(#).xyz() for target residue (ca_pos2), and neighbor reisdue atoms (ca_pos, c_pos, n_pos)
#output: vector of all nine unit vector components [ca_cai, ca_caj, ca_cak, c_cai, c_caj, c_cak, n_cai, n_caj, n_cak]
#bool for passing, (true if all good, false if division by 0 occurs)
def vmu(ca_pos, c_pos, n_pos):
#####################  VECTOR   #####################
	vect = []
	failed = False
	#print ("FUNCTIONS!!!!!!!!!!!!!!!!!*******************!!!!!!!!!!!!!!!!!!!!")
	#pose.residue(neighbors[i]).xyz("CA"), pose.residue(neighbors[i]).xyz("C"), pose.residue(neighbors[i]).xyz("N"), pose.residue(r).xyz('CA'))
	#print (type(ca_pos))
	#print (ca_pos.item(0))
	vect.append(ca_pos.item(0)) 
	vect.append(ca_pos.item(1))
	vect.append(ca_pos.item(2))
	#print ("FUNNER FUNCTIONS!!!!!!!!!!!!!!!!!*******************!!!!!!!!!!!!!!!!!!!!")
	vect.append(c_pos.item(0)-ca_pos.item(0)) 
	vect.append(c_pos.item(1)-ca_pos.item(1)) 
	vect.append(c_pos.item(2)-ca_pos.item(2)) 
	
	
	vect.append(n_pos.item(0)-ca_pos.item(0)) 
	vect.append(n_pos.item(1)-ca_pos.item(1)) 
	vect.append(n_pos.item(2)-ca_pos.item(2)) 
	
	#print ("VECTOR: " , vect)

#####################  MAGNITUDE #####################
	magn = []
	magn1 = sqrt((vect[0]**2) + (vect[1]**2) + (vect[2]**2)) 
	magn2 = sqrt((vect[3]**2) + (vect[4]**2) + (vect[5]**2)) 
	magn3 = sqrt((vect[6]**2) + (vect[7]**2) + (vect[8]**2))
	if (magn1 == 0 or magn2 == 0 or magn3 == 0):
		failed = True #has failed
		uniVec = [0] *9
	else:
		magn.append(magn1)
		magn.append(magn2)
		magn.append(magn3)

		#print ("MAGNITUDE: ", magn)

	##################### UNIT VECTOR #####################
		uniVec = []
		uniVec.append(vect[0]/magn1) 
		uniVec.append(vect[1]/magn1) 
		uniVec.append(vect[2]/magn1)
		uniVec.append(vect[3]/magn2) 
		uniVec.append(vect[4]/magn2) 
		uniVec.append(vect[5]/magn2)
		uniVec.append(vect[6]/magn3) 
		uniVec.append(vect[7]/magn3) 
		uniVec.append(vect[8]/magn3)

	#print ("UNIT VECTOR: ", uniVec)
	return uniVec, failed

#uses dictionary of all hydrophobicities to get the neighbor hydrophobicity
def get_hydrophobicity(seq, position):
	i = position - 1
	aa = seq[i]
	#print ("USING : ", aa)
	if not (  aa in hydrophobicities ):
		print (" ")
		print ("Incorrect amino acid!")
		print ( " ")
		
	return hydrophobicities[aa]

#x_output = open("./TrainingData/x_trainingdata.txt", "w")
#y_output = open("./TrainingData/y_trainingdata.txt", "w")
#testing out all data getters
def gen_binary_off_list(name_in):
	#with workers defaults names as a singl string, fixes by setting names as list containing incoming string
	j = 0
	ns = [name_in]
	for name in ns:
		if not os.path.isfile(  name + "hydrophobicAnd3ShellBinaryX"):
			print ("_*_*_*_*_*_*_*_*_*_*_*_*_*__", name, "_*_*_*__*_*_*_*_*_*_*_*_*_**_*_**_**_*_")	
			try:
				pose = pose_from_pdb(name+".clean.pdb")
				total_len = pose.total_residue() #total protein length
				#total_datapoints = total_len * 293
				#print ("BIN STORED____________________________________________________")
				#bin_stored = np.fromfile("./binaryFiles/"+name+"binaryX", dtype = float, count = -1, sep = "")
			
				#need = True
				'''
				print ("ACTUAL: ", total_datapoints, " STORED: ", bin_stored.shape, " " ,bin_stored.shape[0])
				if not (bin_stored.shape[0] == total_datapoints): #so, failed the check 
					print ("NEEDS FIXING")
					need = True
				else:
					print ("PASSED")
					need = False
				'''
		
				print (pose.pdb_info())
				dist_dict, sphere_counts = getAllDistanceMatrix(pose, 20.0, 30.0, 40.0) #second argument is cutoff for sphere
				#for thing in sphere_counts.keys():
				#	print ("key: ", thing, " in sphere: ", sphere_counts[thing])
				bb_list = getAllbbSASA(pose)
				hb_dict = getAllHBonds(pose)
				xarray = np.zeros((1,356))
				yarray = np.zeros((1,20))
				is_unique = compute_unique_chains(pose)
				failed_residue = False
				seq = pose.sequence()
				#print ("HELLO!")
				for r in range(1, pose.total_residue() + 1):
					if not is_unique[r]:
						continue
					x = []
					#print ("******************************")
					j = j + 1
					res = pose.residue(r)
					#print ("On: ", r , " out of ", pose.total_residue())
				
					x.append(cos(pose.phi(r)))
					x.append(sin(pose.phi(r)))
					x.append(cos(pose.psi(r)))
					x.append(sin(pose.psi(r)))
					x.append(cos(pose.omega(r))) 
					x.append(sin(pose.omega(r)))
					x.append(bb_list[r])
					x.append(getQ3(pose, r))
					x.append(sphere_counts[r][0]) #append count in 30.0 ang sphere
					x.append(sphere_counts[r][1])
					x.append(sphere_counts[r][2])
					neighbors = dist_dict[r] #list of neighbors
					ca_r = pose.residue(r).xyz('CA')
					ca_np = np.array([ca_r.x, ca_r.y, ca_r.z])
					c_r = pose.residue(r).xyz('C')
					c_np = np.array([c_r.x, c_r.y, c_r.z])
					n_r = pose.residue(r).xyz('N')
					n_np = np.array([n_r.x, n_r.y, n_r.z])
					nxp = np.subtract(n_np,ca_np) / np.linalg.norm(np.subtract(n_np,ca_np))
					cxp = np.subtract(c_np, ca_np)/np.linalg.norm(np.subtract(c_np, ca_np))
					xp = -1 *nxp #x axis
					zp = np.cross(cxp, xp)
					zp = zp / np.linalg.norm(zp)
					yp = np.cross(zp,xp)
					yp = yp / np.linalg.norm(yp)
					neg_ca_np = -1*ca_np
					rm = np.transpose(np.array([xp,yp,zp]))
					rm = np.vstack((rm, np.array([0, 0, 0])))
					rm = np.hstack((rm, [[0],[0],[0],[1]]))
					rm = np.matrix(rm)
					tm = np.matrix( [[1, 0, 0, 0], [0, 1, 0, 0],[0,0,1, 0], [-ca_r.x, -ca_r.y, -ca_r.z, 1]])
					m = tm * rm
				
					for i in range(0,15):
					#print ("NEIGHBORS!!!!!!!!!!!!!!!!!*******************!!!!!!!!!!!!!!!!!!!!")
						x.append(cos(pose.phi(neighbors[i])))
						x.append(sin(pose.phi(neighbors[i])))
						x.append(cos(pose.psi(neighbors[i]))) 
						x.append(sin(pose.psi(neighbors[i])))
						x.append(cos(pose.omega(neighbors[i])))
						x.append(sin(pose.omega(neighbors[i])))
						x.append(bb_list[neighbors[i]])
						ca_rsub = pose.residue(neighbors[i]).xyz('CA')
						x.append(sqrt((ca_r.x - ca_rsub.x)**2 + (ca_r.y - ca_rsub.y)**2 + (ca_r.z - ca_rsub.z)**2))
						rsub_ca_raw = pose.residue(neighbors[i]).xyz("CA")
						rsub_ca = np.array([rsub_ca_raw.x, rsub_ca_raw.y, rsub_ca_raw.z, 1])
						rsub_ca = np.array(rsub_ca *m)
						rsub_c = pose.residue(neighbors[i]).xyz("C")
						rsub_c = np.array([rsub_c.x, rsub_c.y, rsub_c.z, 1])
						rsub_c = np.array(rsub_c *m)
						rsub_n = pose.residue(neighbors[i]).xyz("N")
						rsub_n = np.array([rsub_n.x, rsub_n.y, rsub_n.z, 1])
						rsub_n = np.array(rsub_n *m)
						#print ("NEIGHBORS!!!!!!!!!!!!!!!!!*******************!!!!!!!!!!!!!!!!!!!!")
						holder, pass_vmu = vmu(rsub_ca, rsub_c, rsub_n)
						#print ("NEIGHBORS!!!!!!!!!!!!!!!!!*******************!!!!!!!!!!!!!!!!!!!!")
						#if there is not an error in the coordinates
						x.append(holder[0])
						x.append(holder[1])
						x.append(holder[2])
						x.append(holder[3])
						x.append(holder[4])
						x.append(holder[5])
						x.append(holder[6])
						x.append(holder[7])
						x.append(holder[8])
						x.append(getQ3(pose, neighbors[i]))
						x.append(hb_dict[r].count(neighbors[i]))
						#print ("Actual Amino acid is: ", pose.residue(neighbors[i]).name())
						x.append(get_hydrophobicity(seq, neighbors[i])) #get hydro
						x.append(sphere_counts[neighbors[i]][0])
						x.append(sphere_counts[neighbors[i]][1])
						x.append(sphere_counts[neighbors[i]][2])
						if pass_vmu:
							print ("FAILURE DETECTED")
							failed_residue = True
					y = [0] * 20
					order_aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
					#NOTE: pose.residue.aa() -> AA.aa_res for given residue identity
					#print (seq[r-1], " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
					#print ("HOLA!")
					#print (seq)
					if seq[r-1] in order_aa:
						for i in range(0,20):
							if seq[r-1] == order_aa[i]:
								y[i] = 1
					else:
						failed_residue = True
					if not failed_residue:
						xarray = np.vstack((xarray,np.asarray(x)))
						yarray = np.vstack((yarray,np.asarray(y)))
					#print ("reached end of ", r)
				#use numpy vstack and asarray to add rows to a numpy array object
				#print ("old array")
				#print (xarray.shape)
				xarray = numpy.delete(xarray, (0), axis=0)
				yarray = numpy.delete(yarray, (0), axis=0)
				#print ("fixed array")
				#print (xarray.shape)

				if xarray.shape[0] > 0:
					xarray.tofile( name + "hydrophobicAnd3ShellBinaryX", sep ="")
					print ("GOOD______________________________________________________________________________________")
					yarray.tofile( name + "binaryY", sep="")
					#make five-fold cross val files
				else:
					failures.write(name)
					print ("BAD")
					failures.write("\n")

			except:
				print ("FAILURE AT: ", name)
				failures.write(name)
				failures.write("\n")    

def get_hydrophobicity_ALL(seq):
	all_hydr = []
	for aa in seq:
		all_hydr.append(hydrophobicities[aa])
	return all_hydr


#running for single protein:
#1st-create a clean.pdb of the protein
#TODO: GET pro name, run this program
name = "Q59485"
toolbox.cleanATOM(name+".pdb")
#2nd-Run with gen_binary_off_list with single string name input
gen_binary_off_list(name)









	
#print("running program in folder "+ dataPath)
#names_raw = open("finalIDs30PerShort.txt", "r")
#names = names_raw.read().splitlines()
#print (names)
#failures = open("./errorsHydrophobicAndShell.txt", "w")
'''
n = [names[0]]
for name in n:
	pose = toolbox.pose_from_rcsb(name)
	hydrophob = get_hydrophobicity_ALL(pose.sequence())
	for j in range(1,8):
		plt.figure(j)
		dist_dict, sphere_counts = getAllDistanceMatrix(pose, j*10)
		counts = []
		for r in range(1,pose.total_residue()+1):
			counts.append(sphere_counts[r])
		plt.scatter(counts, hydrophob)
		plt.savefig("cutoff" + str(j) + ".png")
'''	



'''
#https://p16.praetorian.com/blog/multi-core-and-distributed-programming-in-python set up multiprocessing with a pool
print (sys.argv)
if __name__ == '__main__':
	with Pool(processes=8) as pool:
		#map doWork to availble Pool processes
		pool.map(gen_binary_off_list, names)

names_raw.close()
failures.close()

#2nd Declare results variables and organization schemes
#x = [cos(phi) sin(phi) cos(psi) sin(psi) cos(omega) sin(omega) bbSASA Q3 15*(cos(phi) sin(phi) cos(psi) sin(psi) cos(omega) sin(omega) bbSASA CaCa_distance Ca_Ca_unitVector Ca_N_unitVector Ca_C_unitVector Q3 Number_Hydrogen_Bonds)]

#y = residue (1X20 vector, 1 one and rest 0's)


#x = [cos(phi) sin(phi) cos(psi) sin(psi) cos(omega) sin(omega) bbSASA Q3 number_residues_in_sphere 15*(cos(phi) sin(phi) cos(psi) sin(psi) cos(omega) sin(omega) bbSASA CaCa_distance Ca_Ca_unitVector Ca_N_unitVector Ca_C_unitVector Q3 Number_Hydrogen_Bonds hydrophobicity(from paper))]
'''
