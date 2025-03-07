#ifndef DECISION_TREE_SINGLE_H
#define DECISION_TREE_SINGLE_H

#include "../math/math_functions.h"
#include <atomic>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>
#include <omp.h>

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

  // Constructor and existing methods
  DecisionTreeSingle(int MaxDepth, int MinLeafLarge, double MinError,
                     int Criteria = 0, int numThreads = 1, int useOmp = 0);
  void train(const std::vector<double> &Data, int rowLength,
             const std::vector<double> &Labels, int criteria = 0);

  void evaluate(const std::vector<double> &X_test, const int rowLength,
                const std::vector<double> &y_test, double &mse_value,
                double &mae_value);
  double predict(const std::vector<double> &Sample) const;
  void saveTree(const std::string &filename);
  void loadTree(const std::string &filename);
  std::map<std::string, std::string> getTrainingParameters() const;
  std::string getTrainingParametersString() const;

  const Tree *getRoot() const { return Root.get(); }
  double getRootMSE() const { return Root ? Root->NodeMetric : 0.0; }
  size_t getRootSamples() const { return Root ? Root->NodeSamples : 0; }

  // this determines at twhich depth to stop creating new threads
  void getMaxSplitDepth() { maxSplitDepth = std::log2(numThreads); }

private:
  std::unique_ptr<Tree> Root;
  int MaxDepth;
  int MinLeafLarge;
  int Criteria;
  double MinError;

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
  int maxSplitDepth = 0;
  std::atomic<int> activeThreads;
  int useOmp = 0;
};

#endif // DECISION_TREE_SINGLE_H
