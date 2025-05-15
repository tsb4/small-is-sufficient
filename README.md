# small-is-sufficient

**small-is-sufficient** is a research project exploring the trade-offs between AI model size and utility for estimating energy savings. 

## Features
The repository contains the following features:

- Provides the code snippets for measuring the energy consumption during inference using CarbonTracker tool.
- Presents a .csv file for each task containing the model utility and model size (number of parameters) for exploring the utility-size tradeoff.

## Tasks

This repository evaluates models for the following tasks:
- Task Generation
- Task Classification
- Image Classification
- Image Segmentation
- Object Detection
- Speech Recognition
- Audio Classification
- Text to Image
- Image-text to Text
- Time Series forecasting


## Project Structure
```
small-is-sufficient/
├── data/ # Datasets and preprocessed files 
├── exps/ # Experiment configurations and scripts
├── exps copy/ # Backup of experiment scripts
├── README.md # Project documentation
└── requirements.txt # Python dependencies
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/tsb4/small-is-sufficient.git
   cd small-is-sufficient
   ```
2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
