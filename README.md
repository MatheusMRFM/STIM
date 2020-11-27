# STIM Execution Guidelines

This repository implements the Spatio-Temporal Influence Maximization (STIM), proposed in "Efficient Information Diffusion in Time-Varying Graphs through Deep Reinforcement Learning". 

There are three folders:

1. **Graphs**: Contains the scripts for generating the graphs used for training and testing. The training and testing graph files are not in this repo. You will have to generate them by running "STIM/Graphs/0_generate_graph.py". To do so, remember to fill the variables defined at the start of the file. By running this script, the graph files for the trianing and test will be generated and saved in folders defined by the TRAIN_FLD and TEST_FLD variables of that same script. The current repo already has the files for the real network used in the paper. The script "STIM/Graphs/1_get_graph_stats.py" generates the stats of the TVGs inside the FOLDER_IN variable, while the script "STIM/Graphs/2_scatter.py" generates the scatter plots of these same TVGs using the output of the previous script;
2. **STIM**: Contains the code for training and testing the STIM model (**main.py**). To run the STIM model, simply run the **main.py** file. Remember to first train the model (after generating the train and test files). The **constants.py** stores all variables used during training. Remember to first set all variables in the latter file;
