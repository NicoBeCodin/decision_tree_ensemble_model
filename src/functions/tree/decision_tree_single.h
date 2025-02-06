#ifndef DECISION_TREE_SINGLE_H
#define DECISION_TREE_SINGLE_H

#include "../math/math_functions.h"
#include <fstream>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>
#include <map>
#include <atomic>
#include <future>

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
  DecisionTreeSingle(int MaxDepth, int MinLeafLarge, double MinError, int Criteria =0, int numThreads =1);
  void train(const std::vector<double> &Data, int rowLength,
             const std::vector<double> &Labels, int criteria = 0);
  double predict(const std::vector<double> &Sample) const;
  void saveTree(const std::string &filename);
  void loadTree(const std::string &filename);
    std::map<std::string, std::string> getTrainingParameters() const;
    std::string getTrainingParametersString() const;


  const Tree *getRoot() const { return Root.get(); }
  double getRootMSE() const { return Root ? Root->NodeMetric : 0.0; }
  size_t getRootSamples() const { return Root ? Root->NodeSamples : 0; }

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
  findBestSplitUsingMAE(const std::vector<double> &Data, int rowLength,
                        const std::vector<double> &Labels,
                        const std::vector<int> &Indices, double CurrentMAE);


//Not necessary for the moment, just tryna merge

                    // preSortFeatures(const std::vector<std::vector<double>> &Data,
                    // const std::vector<int> &Indices);
    


  void serializeNode(const Tree *node, std::ostream &out);
  std::unique_ptr<Tree> deserializeNode(std::istream &in);

  int numThreads = 1;
  std::atomic<int> activeThreads;
};

#endif // DECISION_TREE_SINGLE_H
