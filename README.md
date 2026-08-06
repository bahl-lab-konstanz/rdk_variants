# RDK_VARIANTS
Implementation of the analysis and plots shown in _Eleftheriadis, Garza et al 2027_.

### Project organization
Here is an overview of how to navigate the project
- `figures`: contains all scripts generating raw versions of the figures in the manuscript
- `analysis_behavior`: contains scripts and jupiter notebooks used
- `model`: contains the implementation of the DDM model, together with a lightweight version of a modeling framework under development in the Bahl Lab
- `utils` and `service`: contain useful constants, functions, and utility classes, used to run analysis and compute core quantities appearing in the figures

### Data
Please download the datasets from [here](https://theuselessweb.com/) and populate your parent `data` directory. Keep the organization of the directory unchanged to have the project tree organized consistently to the paths called in the scripts.

### Environment variables
For all the figure-generating scripts to get access to the right path and data, you will need to edit the file 
named `config.env` in the same directory as this README.md file. Then substitute `<full_path_to_data_folder>` with the actual path to the directory where the downloaded and extracted `data` folder is located. Change `<full_path_to_results_folder>` to where you would like to store your figure output files. The folder `results` is usually adjacent to the `data` folder. Both folders need to exist before starting the scripts.


### Dependencies
To install the core dependencies with conda:

- Open terminal
- Navigate to the root directory of this project
- Run the following

```
conda config --set channel_priority flexible
conda env create --file=environment.yaml
conda activate rdk_variants
```

### Running code
Stay in the root folder and run `python -m figures.figure_1`, etc. Note that it is important to run scripts as modules to enable dynamic path resolving. This will generate the raw figures as used in the paper.

### Model simulation and fitting
To simulate an example model, stay in the root folder and run `python -m model.simulate_ddm`.
Before fitting DDM to the data, it is necessary to split the dataset into single-fish files. To do that, run `python -m model.preprocessing`
To fit a model against an example dataset, stay in the root folder and run `python -m model.fit_ddm`. 

Data storage configuration and simulation parameters can be selected inside each of the scripts (check in-line documentation).
