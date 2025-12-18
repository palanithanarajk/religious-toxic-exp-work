# Religious Toxic Classification

This repository contains the implementation of a toxicity classification model specifically focused on religious identity groups. The model utilizes **DeBERTa-v3** with a custom **Hierarchical Attention** mechanism and **Adaptive Pooling** to achieve high performance in multi-label classification of religious toxicity.

## Project Overview

The goal of this project is to accurately identify toxic comments directed towards various religious groups:
- Christian
- Jewish
- Muslim
- Hindu
- Buddhist
- Atheist
- Other Religions

## Key Features

- **Model Architecture**: DeBERTa-v3-base backbone.
- **Hierarchical Attention**: Captures multi-level dependencies in text.
- **Adaptive Pooling**: Combines CLS, Mean, Max, and Attention-weighted pooling for robust feature extraction.
- **Focal Loss**: Handles class imbalance effectively.
- **Threshold Tuning**: Optimizes the classification threshold for maximum F1-score.

## Repository Structure

- `main_notebook.ipynb`: A well-documented Jupyter notebook containing the full pipeline.
- `main.py`: A Python script version of the pipeline.
- `requirements.txt`: List of required Python packages.

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended for training)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd religious-toxic-exp-work
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

The dataset used for this project is from the [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data) Kaggle competition.

Download the `train.csv` (and other related files) from the link above and place it in the project root or specify its path in the code.

The model expects a CSV file (e.g., `train.csv`) with the following columns:
- `comment_text`: The text of the comment.
- `christian`, `jewish`, `muslim`, `hindu`, `buddhist`, `atheist`, `other_religion`: Toxicity labels (0 or 1, or continuous values >= 0.5 for toxicity).

### Running the Re-implementation

#### Using Jupyter Notebook
Open `main_notebook.ipynb` in your preferred environment (JupyterLab, VS Code, etc.) and run the cells sequentially.

#### Using Python Script
Run the script directly from the terminal:
```bash
python main.py
```

## Results

The model achieves significant improvements over baseline CNN/Glove models:

| Metric    | CNN (Glove) - Paper | Custom DeBERTa-V3 |
|-----------|---------------------|-------------------|
| Accuracy  | 95.24               | ~96.97            |
| Precision | 96.59               | ~98.33            |
| Recall    | 96.91               | ~98.72            |
| F1-Score  | 96.75               | ~98.52            |

## License

[Specify License Here]
