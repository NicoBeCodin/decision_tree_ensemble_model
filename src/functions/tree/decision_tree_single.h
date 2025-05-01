#ifndef DECISION_TREE_SINGLE_H
#define DECISION_TREE_SINGLE_H

#include "../math/math_functions.h"
#include <atomic>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <omp.h>
#include <sstream>
#include <tuple>
#include <vector>

class DecisionTreeSingle {
public:
  struct Tree {
    int FeatureIndex = -1;
    double MaxValue = 0.0;
    double Prediction = 0.0;
    bool IsLeaf = false;
    double NodeMetric = 0.0;
    size_t NodeSamples = 0;
    std::unique_ptr<Tree> Left = nullptr;
    std::unique_ptr<Tree> Right = nullptr;
  };
  DecisionTreeSingle() = default;

  // Constructor and existing methods
  DecisionTreeSingle(int MaxDepth, int MinLeafLarge, double MinError,
                     int Criteria = 0, bool useOMP = 0, int numThreads = 1);
  void train(const std::vector<double> &Data, int rowLength,
             const std::vector<double> &Labels, int criteria = 0);

  void evaluate(const std::vector<double> &X_test, const int rowLength,
                const std::vector<double> &y_test, double &mse_value,
                double &mae_value);
  double predict(const double *Sample, int rowLength) const;
  void saveTree(const std::string &filename);
  void loadTree(const std::string &filename);
  std::map<std::string, std::string> getTrainingParameters() const;
  std::string getTrainingParametersString() const;

  const Tree *getRoot() const { return Root.get(); }
  double getRootMSE() const { return RootMSE; }
  int getRootSamples() const { return RootSamples; }

  // this determines at twhich depth to stop creating new threads
  void getMaxSplitDepth() { maxSplitDepth = std::log2(numThreads); }

  std::vector<char> serializeToBuffer();
  static std::unique_ptr<DecisionTreeSingle>
  deserializeFromBuffer(const std::vector<char> &buf);

private:
  std::unique_ptr<Tree> Root;
  int MaxDepth;
  int MinLeafLarge;
  int Criteria;
  int RootSamples;
  bool useOMP;
  double MinError;
  double RootMSE;

  void splitNode(Tree *Node, const std::vector<double> &Data, int rowLength,
                 const std::vector<double> &Labels,
                 const std::vector<int> &Indices, int Depth);

  void splitNodeMAE(Tree *Node, const std::vector<double> &Data, int rowLength,
                    const std::vector<double> &Labels,
                    const std::vector<int> &Indices, int Depth);

  std::tuple<int, double, double>
  findBestSplit(const std::vector<double> &Data, int rowLength,
                const std::vector<double> &Labels,
                const std::vector<int> &Indices, double CurrentMSE);

  std::tuple<int, double, double>
  findBestSplitOMP(const std::vector<double> &Data, int rowLength,
                   const std::vector<double> &Labels,
                   const std::vector<int> &Indices, double CurrentMSE);

  std::tuple<int, double, double>
  findBestSplitUsingMAE(const std::vector<double> &Data, int rowLength,
                        const std::vector<double> &Labels,
                        const std::vector<int> &Indices, double CurrentMAE);

  std::tuple<int, double, double>
  findBestSplitUsingMAEOMP(const std::vector<double> &Data, int rowLength,
                           const std::vector<double> &Labels,
                           const std::vector<int> &Indices, double CurrentMAE);

  // Not necessary for the moment, just tryna merge

  // preSortFeatures(const std::vector<std::vector<double>> &Data,
  // const std::vector<int> &Indices);

  void serializeNode(const Tree *node, std::ostream &out);
  std::unique_ptr<Tree> deserializeNode(std::istream &in);
  

  int numThreads = 1;
  int maxSplitDepth = 2;
  std::atomic<int> activeThreads; // deprecated
};

#endif // DECISION_TREE_SINGLE_H
