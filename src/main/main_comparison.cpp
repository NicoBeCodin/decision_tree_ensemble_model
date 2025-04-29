#include "../pipeline/getModelParameters.h"
#include <sstream>

std::string mpi_setenv(const std::string &var, const std::string &val) {
#ifdef OPEN_MPI // <mpi.h> defines this macro when using Open MPI
  return "-x " + var + "=" + val;
#else // assume MPICH or Intel MPI (MPICH-derived)
  return "-env " + var + " " + val;
#endif
}

int main() {
  std::cout << "Decision Tree Models Comparing Program\n\n";

  int choice;
  std::cout << "Choose an option:\n";
  std::cout << "1. Run individual model\n";
  std::cout << "2. Run all tests\n";
  std::cout << "3. View models comparison\n";
  std::cin >> choice;

  switch (choice) {
  case 1: {
    std::cout << "\nChoose model to use:\n";
    std::cout << "1. Single Decision Tree\n";
    std::cout << "2. Bagging\n";
    std::cout << "3. Boosting\n";

    int model_choice;
    std::cin >> model_choice;

    std::string parameters = std::to_string(model_choice);
    getModelParameters(model_choice, parameters);

    //     int mpi_procs = 0; // 0 ⇒ no MPI
    //     if (model_choice == 2) {
    //       std::cout << "Run with MPI? 0 = no MPI, N = number of processes: ";
    //       std::cin >> mpi_procs;
    //     }

    //     std::string mpi_cmd;

    // if (model_choice == 2 && mpi_procs > 0) {
    //     auto env = [](std::string var, std::string val) {
    // #ifdef OPEN_MPI          // defined by <mpi.h> when using Open MPI
    //         return "-x " + var + "=" + val;
    // #else                    // MPICH, Intel-MPI, MS-MPI…
    //         return "-env " + var + " " + val;
    // #endif
    //     };

    //     int threadsPerRank =   std::thread::hardware_concurrency() /
    //     mpi_procs; threadsPerRank = std::max(1, threadsPerRank);
    //     // params.numThreads   = threadsPerRank;          // keep Bagging in
    //     sync

    //     mpi_cmd  = "mpiexec -n " + std::to_string(mpi_procs) + " "
    // #ifdef OPEN_MPI
    //              + "--map-by socket --bind-to core "
    // #endif
    //              + env("OMP_NUM_THREADS", std::to_string(threadsPerRank)) + "
    //              "
    //              + env("OMP_PROC_BIND",  "close") + " "
    //              + env("OMP_PLACES",     "cores") + " "
    //              + "./MainEnsemble " + parameters;
    // } else {
    //     mpi_cmd = "./MainEnsemble " + parameters;
    // }

    // std::cout << "Executing: " << mpi_cmd << '\n';
    // std::system((mpi_cmd + " 2>&1").c_str());
    int mpi_procs = 0; // 0 ⇒ run without mpiexec
    if (model_choice == 2) {
      std::cout << "Run with MPI? 0 = no MPI, N = number of processes [0]: ";
      std::cin>>mpi_procs;
    
    }

    auto env = [](const std::string &var, const std::string &val) {
#ifdef OPEN_MPI // set by <mpi.h> when compiled with Open-MPI
      return "-x " + var + "=" + val;
#else // MPICH / Intel-MPI / MS-MPI
      return "-env " + var + " " + val;
#endif
    };

    std::string mpi_cmd;
    if (model_choice == 2 && mpi_procs > 0) {
      /* ── ask for OpenMP customisation ──────────────────────────────────── */
      const unsigned hwCores = std::thread::hardware_concurrency();
      unsigned omp_threads = std::max(1u, hwCores / mpi_procs);

      std::cout << "OpenMP threads per rank          [" << omp_threads << "]: ";
      std::string tmp;
      std::getline(std::cin, tmp);
      if (!tmp.empty())
        omp_threads = std::stoi(tmp);

      std::cout << "OMP_PROC_BIND (none/close/spread) [close]: ";
      std::string omp_bind;
      std::getline(std::cin, omp_bind);
      if (omp_bind.empty())
        omp_bind = "close";

      std::cout << "OMP_PLACES   (cores/threads)      [cores]: ";
      std::string omp_places;
      std::getline(std::cin, omp_places);
      if (omp_places.empty())
        omp_places = "cores";

      std::cout << "Bind MPI ranks to cores? (y/n)    [y]: ";
      std::string bindRanks;
      std::getline(std::cin, bindRanks);
      bool doBind =
          bindRanks.empty() || bindRanks[0] == 'y' || bindRanks[0] == 'Y';

    
    std::cout<< "Change the num_threads inside of program to match the OMP_NUM_THREADS ? [yes]: ";
    std::string changeNumThreads;
      std::getline(std::cin, changeNumThreads);
      if (omp_places.empty())
        omp_places = "yes";
    //Change the values
    if (changeNumThreads=="yes"){
        auto lastSpace = parameters.find_last_of(' ');
        if (lastSpace == std::string::npos) {
            /* the string contains only one field – append a space+token    */
            parameters += ' ' + std::to_string(omp_threads);
        } else {
            parameters.replace(lastSpace + 1, std::string::npos,
                std::to_string(omp_threads));
            }
            
        }
      /* ── build the mpiexec command ────────────────────────────────────── */
      std::ostringstream ss;
      ss << "mpiexec -n " << mpi_procs << " ";
#ifdef OPEN_MPI
      if (doBind)
        ss << "--map-by socket --bind-to core ";
#endif
#ifdef MPICH_VERSION
      if (!doBind)
        ss << "--bind-to none ";
#endif
      ss << env("OMP_NUM_THREADS", std::to_string(omp_threads)) << " "
         << env("OMP_PROC_BIND", omp_bind) << " "
         << env("OMP_PLACES", omp_places) << " " << "./MainEnsemble "
         << parameters;

      mpi_cmd = ss.str();
    } else {
      mpi_cmd = "./MainEnsemble " + parameters;
    }

    std::cout << "\nExecuting: " << mpi_cmd << "\n\n";
    std::system((mpi_cmd + " 2>&1").c_str());
    break;
  }
  case 2: {
    std::cout << "\nRunning all tests...\n\n";

    std::cout << "=== Math Functions Tests ===\n";
    system("./math_functions_test");

    std::cout << "\n=== Decision Tree Tests ===\n";
    system("./decision_tree_test");

    std::cout << "\n=== Bagging Tests ===\n";
    system("./bagging_test");

    std::cout << "\n=== Boosting Tests ===\n";
    system("./boosting_test");

    std::cout << "\n=== Cross Validation Tests ===\n";
    system("./cross_validation_test");

    std::cout << "\nAll tests completed.\n";
    break;
  }
  case 3: {
    std::cout << "\nDisplaying previous results...\n";
    std::ifstream file("../results/all_models_comparison.md");
    if (file.is_open()) {
      std::string line;
      while (std::getline(file, line)) {
        std::cout << line << '\n';
      }
      file.close();
    } else {
      std::cout << "No previous results found. Please run tests first.\n";
    }
    break;
  }
  default:
    std::cout << "Invalid option\n";
    return 1;
  }

  return 0;
}
