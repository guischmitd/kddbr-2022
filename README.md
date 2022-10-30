# KDDBR-2022 Submission Repo

Solution for the 2022 KDDBR kaggle [competition](https://www.kaggle.com/competitions/kddbr-2022).

Author: Guilherme Baraúna

https://www.kaggle.com/guischmitd


---
## Running the code
To run the solution, follow these steps:

1. Download the competition data from kaggle and unzip it into `data/raw` to replicate the following structure:
```
data
├── feats
├── models
├── processed
├── raw
│   ├── test
│   │   └── test
│   └── train
│       └── train
└── subs
```

2. Create a virtual environment, activate it and install dependencies:
```bash
# On Linux/MacOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# On Windows (Git Bash or PowerShell)
python3 -m venv .venv
source .venv/Scripts/activate # Or activate.ps1
pip install -r requirements.txt
```

3. Run all cells on `kddbr/final_submission.ipynb`. Each section of the notebook will generate specific files required for next cells. 

> Note: Since some of the processes take some time, files will be cached by default and accessed in case they are found. You can skip loading the cached data by setting `FORCE_RUN=True` on specific cells.

4. Models will be saved under `data/models` as joblib pickled files. Submissions in the standard competition format (.csv) will be sabe under `data/subs`
