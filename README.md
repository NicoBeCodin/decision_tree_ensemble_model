# Decision Tree Ensemble Model

## Overview
A C++ library implementing and benchmarking five supervised learning methods:
1. **DecisionTree**: a single CART decision tree implemented from scratch, supporting mean squared and absolute error criteria with optional OpenMP parallelism.
2. **Bagging**: bootstrap aggregation of multiple decision trees to reduce variance, with configurable number of trees and tree hyperparameters.
3. **Boosting**: a custom gradient boosting implementation that sequentially trains weak learners to minimize a specified loss, featuring early stopping and learning rate control.
4. **LightGBM**: integration of Microsoft’s LightGBM library for fast, histogram-based gradient boosting with support for large datasets.
5. **AdvancedGBDT**: a custom GBDT variant with DART-style dropout and flexible binning methods (quantile or frequency), for improved regularization and performance.

A Python benchmarking script and plotting utilities automate experiments and generate performance graphs.

---

## Prerequisites

### Common
- A C++17‑capable compiler (Clang or GCC).
- [CMake](https://cmake.org/) 3.10 or higher.
- [Graphviz](https://graphviz.org/) (for tree visualizations).
- Python 3.8+ and `pip`.

### Linux (Debian/Ubuntu)
```bash
apt-get update
apt-get install cmake build-essential libomp-dev graphviz python3 python3-pip lightgbm
```

### macOS (Homebrew)
```bash
brew update
brew install cmake libomp graphviz python3 lightgbm
```

### Python
```bash
python3 -m pip install --user -r requirements.txt
# Plotting dependencies:
python3 -m pip install matplotlib pandas
# LightGBM bindings:
python3 -m pip install lightgbm
```

### Environment Variables
- `OMP_NUM_THREADS`: controls the number of threads for OpenMP-enabled models.
- `USE_MPI` (CMake flag): enable MPI-based parallelism for Bagging.

---

## Building the C++ Project

```bash
mkdir build && cd build
# enable or disable OpenMP
cmake -DOPENMP=ON ..
# To build with MPI support (for Bagging):
cmake -DUSE_MPI=ON -DOPENMP=ON ..
make
```

This produces three executables in `build/`:
- `DataClean` (CSV preprocessing)
- `MainEnsemble` (single-run benchmarking)
- `MainKFold` (k‑fold cross‑validation)

---

## Usage Examples

### Data Cleaning
```bash
./DataClean ../data/raw.csv ../data/clean.csv
```

### Single Experiment Suite
```bash
./MainEnsemble [OPTIONS]
```
Launches an interactive menu to choose one of five methods and optionally override hyperparameters via command‑line flags, e.g.:  
```bash
./MainEnsemble 2 --n_estimators=100 --max_depth=10 --use_omp=1
```

#### Common Command-Line Flags
- `--n_estimators=<int>`: number of trees/estimators.
- `--max_depth=<int>`: maximum tree depth.
- `--learning_rate=<float>`: shrinkage rate for boosting.
- `--use_omp=<0|1>`: disable (0) or enable (1) OpenMP parallelism.
- `--min_data_leaf=<int>`: minimum samples per leaf for AdvancedGBDT.
- `--num_leaves=<int>`: max leaf count for LightGBM.

### Hyperparameter Details

#### DecisionTree
- `--max_depth=<int>` (default: 60): Maximum depth of the tree.
- `--min_samples_split=<int>` (default: 2): Minimum number of samples required to split an internal node.
- `--min_impurity_decrease=<float>` (default: 1e-12): Minimum impurity decrease required to split a node.
- `--use_split_histogram=<0|1>` (default: 0): Enable (1) or disable (0) histogram-based splitting.
- `--use_omp=<0|1>` (default: 0): Enable (1) or disable (0) OpenMP parallelism.
- `--num_threads=<int>` (default: 1): Number of threads to use when OpenMP is enabled.

#### Bagging
- `--n_estimators=<int>` (default: 20): Number of trees to aggregate.
- `--max_depth=<int>` (default: 60): Maximum depth for each base tree.
- `--min_samples_split=<int>` (default: 2): Minimum samples to split a node in base trees.
- `--min_impurity_decrease=<float>` (default: 1e-6): Impurity threshold for splitting in base trees.
- `--which_loss_function=<0|1>` (default: 0): Loss function for aggregation: 0=MSE, 1=MAE.
- `--use_split_histogram=<0|1>` (default: 0): Enable histogram splitting in base trees.
- `--use_omp=<0|1>` (default: 0): Enable OpenMP parallelism.
- `--num_threads=<int>` (default: 1): Number of threads per tree when using OpenMP.

#### Boosting (Custom)
- `--n_estimators=<int>` (default: 75): Number of boosting iterations.
- `--learning_rate=<float>` (default: 0.07): Shrinkage rate of each new tree.
- `--max_depth=<int>` (default: 15): Maximum depth of each weak learner.
- `--min_samples_split=<int>` (default: 3): Minimum samples to split nodes in weak learners.
- `--min_impurity_decrease=<float>` (default: 1e-5): Impurity threshold for splitting in weak learners.
- `--which_loss_function=<0|1>` (default: 0): Loss function: 0=MSE, 1=MAE.
- `--use_split_histogram=<0|1>` (default: 1): Enable histogram-based splitting.
- `--use_omp=<0|1>` (default: 0): Enable OpenMP parallelism.
- `--num_threads=<int>` (default: 1): Threads per iteration when using OpenMP.

#### LightGBM
- `--n_estimators=<int>` (default: 100): Number of boosting rounds.
- `--learning_rate=<float>` (default: 0.1): Learning rate (shrinkage).
- `--max_depth=<int>` (default: -1): Maximum tree depth (-1 for no limit).
- `--num_leaves=<int>` (default: 31): Maximum leaves per tree.
- `--subsample=<float>` (default: 1.0): Fraction of data to use per iteration.
- `--colsample_bytree=<float>` (default: 1.0): Fraction of features to use.

#### AdvancedGBDT
- `--n_estimators=<int>` (default: 200): Number of trees.
- `--learning_rate=<float>` (default: 0.01): Learning rate.
- `--max_depth=<int>` (default: 50): Maximum depth per tree.
- `--min_data_leaf=<int>` (default: 1): Minimum data per leaf.
- `--num_bins=<int>` (default: 1024): Number of bins for feature histograms.
- `--use_dart=<0|1>` (default: 1): Enable DART dropout technique.
- `--dropout_rate=<float>` (default: 0.5): Dropout rate for DART.
- `--skip_drop_rate=<float>` (default: 0.3): Skip-drop probability for DART.
- `--binning_method=<0|1>` (default: 1): Binning method: 0=Quantile, 1=Frequency.

### K‑Fold Cross‑Validation
```bash
./MainKFold
```
Select a method and number of folds to run cross‑validation.

---

## Python Benchmarking & Plotting

From the project root:
```bash
cd script
python3 benchmark.py   # runs experiments, writes CSV
cd ../plots
python3 plot.py        # reads CSV and generates figures
```

- `benchmark.py` writes its results to `script/benchmark_results_extended.csv`.
- `plot.py` reads the CSV and outputs figures into `plots/figures/` as PNG files.

---



## Project Structure
```
/decision_tree_ensemble_model
├─ CMakeLists.txt
├─ src/            # C++ source (models, utilities, pipelines)
├─ build/          # build artifacts
├─ script/         # Python script to run batch experiments and export CSV
├─ plots/          # Python script and output directory for generated figures
└─ README.md       # project overview and instructions
```



## **Usage Tips**

- **Data Paths**: Ensure that datasets are available in the correct path (`../datasets/`).
- **Graphviz Installation**: If you encounter errors about missing `dot`, install **Graphviz** with:
  ```bash
  sudo apt-get install graphviz
  ```
  **Note:** If errors persist, make sure `dot` is in your system's PATH. You can verify this by running:
  ```bash
  which dot
  ```
  If it is not found, you may need to add Graphviz to your PATH or specify the full path to the `dot` binary.

- **Execution Errors**: If errors like `command not found` occur, ensure you are running the executables from the `build/` directory.
- **Permissions**: If you encounter permission issues when running the executables, you may need to set the executable bit with:
  ```bash
  chmod +x DataClean MainEnsemble MainKFold
  ```

---

## License / Ownership

All code and materials produced during this project as part of the Université de Versailles Saint-Quentin-en-Yvelines (UVSQ) curriculum are the property of UVSQ. All rights reserved.
