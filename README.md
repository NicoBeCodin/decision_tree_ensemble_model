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