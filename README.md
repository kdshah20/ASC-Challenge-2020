# ASC-Challenge-2020

This repository was created as a part of submission for American Society of Composites (ASC) Simulation Challenge 2020. We work on the problem of developing a data-driven technique to predict tensile strength of 8-ply unidirectional composites of any given stacking sequence and to optimize stacking sequences for a desired strength.

We submit in this repository files created for solving the forward problem of predicting tensile strength of a unidirectional composite given a stacking sequence. The reverse problem of optimizing stacking sequence for a desired strength is difficult and an area of future work.

We use a neural network to solve the forward problem of predicting strength based on a given stacking sequence. The data required for training the neural network is generated via a parametric study performed using FE element software Abacus. A python script is then used to extract force-time curves from Abacus simulation results to find the ultimate tensile strength and effective stiffness of a given 8-ply composite laminate sequence.

The extracted data is then pre-processed in python before giving it as an input to the neural network. Tensorflow module in python is used to create a neural network with 9 input features, 1 hidden layer with 9 nodes and 1 output layer. Stacking sequence of each laminate and the effective laminate stiffness are used as input features.

We also submit our data files for 8-ply composite laminates (Stacking sequences and strength) that were used in our model. Addionally we also submit data for 8-ply open hole composite laminates to be used in our future work and a script for determing the strength for any given orientation of a open-hole lamina specimen.

The list below gives the name of each file with its description to help the reader in recreating our analysis. 

1) Tension-parametric-asc.inp: Abacus file for parametric study
2) abaq_parametric_script.psf: Abacus file for parametric study
3) extract_force_time.py: Python script to extract force-time curves from Abacus
4) Eff_stiffness.py: Obtain effective stiffness of 8-ply laminate based on a given stacking sequence
5) integrating_data.py: Python script for integrating data from Abacus results and setting it up to be used as an input for neural network
6) NN_angle.py: Python script for creating, fitting and evaluating a neural network based on the input data
7) stackseq.txt: Data file contaning 1314 different 8 ply composite laminate stacking sequences
8) strength_values.txt: Data file containing corressponding tensile strength values for stacking sequences in stackseq.txt
9) Effective_stiffness.txt: Data file for effective stiffness values for 8-ply composite laminate specimens.
10) theta_values-ohT: Data file contaning 8 ply open hole composite laminate stacking sequences
11) strength_values-ohT: Data file contaning 8 ply open hole composite laminate tensile strength values for stacking sequences in theta_values-ohT
12) effective-Ex-ohT: Data file for effective laminate stiffness for open hole composite specimens
13) openhole_v2.py: Python script for determining analytical strength of any given open-hole lamina orientation.
