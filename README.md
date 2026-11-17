# Mitigating-Spurious-Background-Correlations-in-Fine-Grained-Bird-Classification-via-Group-Distributionally-Robust-Optimization

This repository contains the implementation, evaluation pipeline, and Grad-CAM explainability analysis comparing Empirical Risk Minimization (ERM) and Group Distributionally Robust Optimization (Group DRO) on the WILDS Waterbirds benchmark using an ImageNet-pretrained ResNet-50 backbone.

Repository Structure
├── waterbirds_erm_vs_gdro.ipynb   # End-to-end Google Colab notebook (training, evaluation, Grad-CAM)
├── report/
│   ├── es678_project_report.tex   # LaTeX source for the 4-page ES 678 project report
│   └── neurips_2026.sty           # Course formatting style file
├── results/                       # Saved training curves, comparison tables, and Grad-CAM plots
└── README.md                      # Setup, data access, and reproduction instructions

Data Access Instructions
This project uses the Waterbirds dataset from the WILDS benchmark (wilds Python package).
Automatic Download: You do not need to download the dataset manually. Running the data-loading cell in the notebook executes get_dataset(dataset="waterbirds", download=True), which automatically fetches and extracts the dataset (~1.1 GB) from the official WILDS repository into ./data/.
Official Split Details:
Train: 4,795 images (95% majority groups: Waterbird on Water, Landbird on Land; 5% minority groups)
Validation: 1,199 images (balanced across backgrounds within each class; used exclusively for checkpoint selection)
Test: 5,794 images (balanced across backgrounds within each class)
Access Restrictions: The dataset is publicly available under the WILDS benchmark license (constructed from Caltech-UCSD Birds-200-2011 and Places) and requires no API keys or private credentials.

Setup & Rebuild Steps
Option A: Run in Google Colab (Recommended)
Open waterbirds_erm_vs_gdro.ipynb in Google Colab.
Enable GPU acceleration via Runtime  Change runtime type  T4 GPU.
Run all cells sequentially from top to bottom (Runtime  Run all).
The notebook automatically installs wilds and opencv-python-headless, mounts Google Drive to save model checkpoints and figures, and fixes random seeds (seed = 42) for reproducibility.
Option B: Run Locally
Clone the repository:
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision wilds numpy matplotlib pandas opencv-python-headless tqdm jupyter
Launch Jupyter and run the notebook:
jupyter notebook waterbirds_erm_vs_gdro.ipynb
(Note: When running locally outside Colab, comment out from google.colab import drive and set save_dir = "./outputs".)
Rebuilding the LaTeX Report
To compile the 4-page PDF report from source:
cd report
pdflatex es678_project_report.tex
pdflatex es678_project_report.tex

Key Results Summary
Method
Average Test Accuracy
Worst-Group Test Accuracy (Waterbird on Land)
ERM (Baseline)
87.88%
69.00%
Group DRO
87.28%
77.38% (+8.38%)



