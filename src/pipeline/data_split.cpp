#include "data_split.h"

bool splitDataset(DataParams& data_params) {
    int mpiRank=0;  
    #ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    #endif

    DataIO data_io;
    
    data_params.rowLength = 0; // Initialize
    auto [X, y] = data_io.readCSV("../datasets/processed/cleaned_data.csv", data_params.rowLength);
    
    if (X.empty() || y.empty()) {
      std::cerr << "Unable to open the data file, please check the path." << std::endl;
      return false;
    }
  
    if(mpiRank==0){
      
          std::cout << "X size : " << X.size() << std::endl;
          std::cout << "y size : " << y.size() << std::endl;
    }
    // Creates saved models folder if non existant
    createDirectory("../saved_models");
  
    // We resize rowLength because that it the size of a data row without label
    data_params.rowLength = data_params.rowLength - 1;
    size_t train_size = static_cast<size_t>(y.size() * 0.8) * data_params.rowLength;
  

    if(mpiRank==0){
      std::cout << "Train size : " << train_size << std::endl;
    }
  
    data_params.X_train.assign(X.begin(), X.begin() + train_size);
    data_params.y_train.assign(y.begin(), y.begin() + train_size / data_params.rowLength);
    data_params.X_test.assign(X.begin() + train_size, X.end());
    data_params.y_test.assign(y.begin() + train_size / data_params.rowLength, y.end());
    if(mpiRank==0){

      std::cout << "X_train size : " << data_params.X_train.size() << std::endl;
      std::cout << "y_train size : " << data_params.y_train.size() << std::endl;
      std::cout << "X_test size : " << data_params.X_test.size() << std::endl;
      std::cout << "y_test size : " << data_params.y_test.size() << "\n" << std::endl; 
    }
    return true;
}