param(
  [string]$PythonExe = "py -3.7",
  [string]$Subjects = "1-30",
  [string]$Task = "left_right",
  [int]$Epochs = 35
)

$ErrorActionPreference = "Stop"

Write-Host "Installing dependencies for Python 3.7..."
Invoke-Expression "$PythonExe -m pip install --upgrade pip"
Invoke-Expression "$PythonExe -m pip install -r requirements_py37.txt"

Write-Host "Downloading EDF files..."
Invoke-Expression "$PythonExe scripts/MIND_Get_EDF.py --subjects $Subjects --runs 4,6,8,10,12,14 --out-dir data/raw_edf"

Write-Host "Preparing CSV dataset..."
Invoke-Expression "$PythonExe scripts/prepare_dataset.py --raw-dir data/raw_edf --out-dir data/processed/EEG-Motor-Movement-Imagery-Dataset --subjects $Subjects --runs 4,6,8,10,12,14 --task $Task"

Write-Host "Training baselines and hybrid model..."
Invoke-Expression "$PythonExe train_compare.py --data-dir data/processed/EEG-Motor-Movement-Imagery-Dataset --results-dir results --models all --epochs $Epochs"

Write-Host "Done. Open the dashboard with:"
Write-Host "$PythonExe -m streamlit run ui_dashboard.py"

