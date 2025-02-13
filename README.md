# **Decision Tree Ensemble Model**

This project implements several machine learning methods, including **Decision Trees**, **Bagging**, **Boosting**, and **XGBoost**. Additionally, it provides a **data cleaning utility** to preprocess datasets for use in these models.

---

## **How to Build the Project**

To generate the executables (**DataClean**, **MainEnsemble**, and **MainKFold**), follow these steps:

1. **Install CMake** (if not already installed):
   ```bash
   sudo apt-get update
   sudo apt-get install cmake
   ```
2. **Create and navigate to the build directory**:
   ```bash
   mkdir build
   cd build
   ```
3. **Run CMake to configure the project**:
   ```bash
   cmake ..
   #or if you want to disable OpenMP optimizations
   cmake -DOPENMP=OFF ..
   ```
4. **Compile the project**:
   ```bash
   make
   ```
5. The following executables will be available in the `build/` directory:
   - **DataClean**
   - **MainEnsemble**
   - **MainKFold**

---

## **How to Run the Executables**

### **DataClean**
Run the **DataClean** executable with two arguments:
```bash
./DataClean <inputpathfile> <outputpathfile>
```
Example:
```bash
./DataClean /path/to/file/input.csv /path/to/file/output.csv
```

---

### **MainEnsemble**
Run the **MainEnsemble** executable and choose one of the four available methods:
```bash
./MainEnsemble
```
Options available:
```
1: Simple Decision Tree
2: Bagging
3: Boosting
4: XGBoost
```
After selecting a method, the program will train the model and display performance metrics like training time, evaluation time, and mean squared error (MSE) or mean absolute error (MAE).

---

### **MainKFold**
Run the **MainKFold** executable and choose one of the three available methods:
```bash
./MainKFold
```
Options available:
```
1: Simple Decision Tree
2: Bagging
3: Boosting
```
The program performs K-Fold cross-validation for each method and displays the performance metrics.

---



### **Decision Tree Models Comparison**

Run the **Decision Tree Models Comparison** executable and choose from a variety of options to either run individual models, run all available tests, or view previous model comparisons.

```bash
./decision_tree
```

Options available:
```
1: Run individual model
2: Run all tests
3: View models comparison
```

After selecting an option, you will be prompted with the following steps:

#### **Option 1: Run Individual Model**

1. Choose the model you want to use:
   ```
   1: Single Decision Tree
   2: Bagging
   3: Boosting
   4: XGBoost
   ```

2. After selecting a model, you will be asked whether to load an existing tree model:
   ```
   Would you like to load an existing tree model? (1 = Yes (currently unused), 0 = No): 
   ```

3. If you choose **No**, you will then have the option to customize the parameters for the selected model:
   ```
   Do you want to customize parameters? (1 = Yes, 0 = No): 
   ```

   - If you choose **Yes**, you will be asked for various parameters specific to the model selected, such as:
     - **Maximum depth** of the tree
     - **Minimum samples** required to split a node
     - **Splitting criteria** (MSE or MAE)
     - **Learning rate** (for Boosting and XGBoost)

   - If you choose **No**, the program will use the default parameters.

4. Once the parameters are set, the program will train the selected model and display relevant performance metrics such as training time, evaluation time, and error metrics (MSE or MAE).

#### **Option 2: Run All Tests**

1. This option runs a suite of tests for all available models:
   ```
   === Math Functions Tests ===
   === Decision Tree Tests ===
   === Bagging Tests ===
   === Boosting Tests ===
   === XGBoost Tests ===
   === Cross Validation Tests ===
   ```

2. After running the tests, performance metrics will be displayed for each test case.

#### **Option 3: View Models Comparison**

1. This option will display the results of previous model comparisons stored in the `all_models_comparison.md` file.
   
   If no previous results are available, the program will prompt:
   ```
   No previous results found. Please run tests first.
   ```

---


## **CMake Configuration**

Here is a simplified version of the **CMakeLists.txt** configuration:

```cmake
cmake_minimum_required(VERSION 3.10)
project(DecisionTreeEnsembleModel)

set(CMAKE_CXX_STANDARD 17)

# Add subdirectories for submodules
add_subdirectory(functions_io)
add_subdirectory(functions_tree)
add_subdirectory(ensemble_bagging)
add_subdirectory(ensemble_boosting)
add_subdirectory(ensemble_boosting_XGBoost)
add_subdirectory(data_clean)

# Create executables
add_executable(DataClean main_data_clean.cpp)
add_executable(MainEnsemble main.cpp)
add_executable(MainKFold main_kfold.cpp)

# Link libraries to the executables
target_link_libraries(DataClean Data_Clean)

target_link_libraries(MainEnsemble
    FunctionsIO
    FunctionsTree
    Bagging
    Boosting
    Boosting_XGBoost
)

target_link_libraries(MainKFold
    FunctionsIO
    FunctionsTree
    Bagging
    Boosting
)
```

This file defines how the executables are built and the libraries they link to.

---

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
