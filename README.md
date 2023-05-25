# P8 - Image Classification with Learned Class Embeddings
Project description

# Installation
The project has been tested with Python 3.9.16 on both Windows 10 and Arch Linux.

1. Cloning the repository.
```shell
git clone https://github.com/Simonbolsen/P8.git
cd .\P8\
```

2. Install pytorch from [here](https://pytorch.org/get-started/locally/). We recommend using CUDA for faster training.
3. (Windows Only) Get Microsoft Visual C++ 14.0 or greater through the Microsoft C++ Build Tools [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/). You must all select Windows 10 SDK.
4. Install remaining dependencies
```shell
pip install -r requirements.txt
```

You can try running the following test experiment

```shell
python ADD ARGS FOR TEST
```

# Running experiments
All experiments are located in `.\.vscode\tasks.json`. The experiments are categorised as follows:
- CL: Ray experiments
- BEST: Embeddings extraction

Running CL experiments will create a directory at `~\ray_results\` with results. When running tasks labelled as BEST, embeddings will be saved in the `.\embeddingData` directory. Be warned that embedding data may require ~30GB of space per run.

# Extract data and plots
1. Create the folder `.\plotData\`
2. Add experiment folder name from `~\ray_results\` to the bottom of `.\src\embeddingAnalysis\article_vizualization.py`
3. Run the script
```
python .\src\embeddingAnalysis\article_vizualization.py
```






