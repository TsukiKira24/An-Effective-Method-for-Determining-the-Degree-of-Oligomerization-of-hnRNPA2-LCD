FOR MONOMER MD SIMULATION ANALYSIS
1. Molecular dynamics simulations were analyzed for monomeric protein systems 
under three pH conditions (4.0, 7.4, and 8.5). The analyzed trajectories were
converted into pdb files and consisted of frames sampled every 1 ns. The pdb
files were renamed to follow FOLD{runs_number}_pH{pH_reference_number}_ion_1ns.pdb 
naming pattern (example: FOLD1_pH40_ion_1ns.pdb). 
2. Files thats summarize all of the protein-ions interactions were generated for 
every file.
python monomer_get_protein_n_Cl-_Na+_interactions.py
3. If there is more than one atoms interaction between residue and ion (within
> 5 A detected), it is count only as one interaction in the frame. 
