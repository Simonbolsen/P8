# P8 - A Geometric Approach to Image Classification using Neural Networks with Class Embeddings
The Moving Targets Model uses class embeddings to represent classes as d-dimensional vectors. The vectors are learned during training, and a nearest neighbour classifier is used for classification. This repository contains experiments and data generation scripts for five loss functions over a total of 75 experiments.

# Installation
The project has been tested with Python 3.9.16 on both Windows 10/11 and Arch Linux.

1. Cloning the repository.
```shell
git clone https://github.com/Simonbolsen/P8.git
cd .\P8\
```

2. Install pytorch from [here](https://pytorch.org/get-started/locally/). We recommend using CUDA for faster training.
3. Install remaining dependencies
```shell
pip install -r requirements.txt
```

# Running experiments
All experiments are located in `.\.vscode\tasks.json`. The experiments are categorised as follows:
- CL: Ray experiments
- BEST: Embeddings extraction

Running CL experiments will create a directory at `~\ray_results\` with results. You may need to modify the requested CPU and GPU requirements for each experiment. When running tasks labelled as BEST, embeddings will be saved in the `.\embeddingData\` directory. Be warned that embedding data may require ~30GB of space per run.

# Extract data and plots
1. Create the folder `.\plots\plotData\`
2. Add experiment folder names from `.\embeddingData\` directory to the bottom of `.\src\embeddingAnalysis\article_vizualization.py` inside `input_folders`
3. Run the script
```
python .\src\embeddingAnalysis\article_vizualization.py
```






