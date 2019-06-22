# ProteinDesignDNNs

Three pre-trained DNNs for Inverse Protein Problem type predictions (that is, predicting an amino acid sequence which will fold into a desired structures).  These networks take inputs of a single site in the protein structure, and output a 1x20 vector of probabilities which are the likelihood of the 20 common amino acids occupying that site in the structure.  

PUT A TABLE HERE WITH THREE NETWORKS AND CROSS VALIDATION ACCURACY

Two of the networks (NAME AND NAME)use only backbone-based information to make predictions, these networks can be used to generate a probability profile for every site in a desired structure to determine the most sequence.  NAME is a replication of Wang et al's work (CITATION) which these networks were inspired by.  NAME is similar to replicated, but with additional inputs and the addition of dropout and batch normalization network layers.

The third network, NAME, uses both backbone and side-chain information to make prediction.  Therefore, it cannot be used by iteself to generate an entire amino acid sequence, but can instead be used to refine individual positions in a sequence.  More importantly, it can be used to help plan single site mutation experiments.  

To use the pre-trained networks:
1) Download a .pdb file or create the desired structure and save it to a .pdb format 
2) Run NAME.PY with the file name as input in NAMEFUNCTION.  (If running multiple files, use NameFunction in NAME.PY and input a .txt file with your pdb file names as input)
3)Double check that binary files were created for your structure
4) Set the correct structural name in NAME2.PY to get predictions and generate output graphs


