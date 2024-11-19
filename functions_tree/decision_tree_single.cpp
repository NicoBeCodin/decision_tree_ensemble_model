//decision_tree_single.cpp
//Yifan
//14.11

#include <limits>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include "decision_tree_single.h"

    /**
     * @brief Constructeur : initialise la profondeur maximale, la taille minimale de feuille, 
     * l'erreur minimale et le critère de division.
     */
DecisionTreeSingle::DecisionTreeSingle(int MaxDepth, int MinLeafLarge, double MinError, SplittingCriteria* Criteria)
    : MaxDepth(MaxDepth), MinLeafLarge(MinLeafLarge), MinError(MinError), Criteria(Criteria), Root(nullptr) {}

    /**
     * @brief Fonction d'entraînement pour l'arbre de décision.
     */
void DecisionTreeSingle::train(const std::vector<std::vector<double>>& Data, const std::vector<double>& Labels) {
    Root = std::make_unique<Tree>(); // Créer un nouvel arbre et l'assigner au pointeur Root.
    std::vector<int> Indices(Data.size()); // Initialiser les indices à la taille des données.
    std::iota(Indices.begin(), Indices.end(), 0); // Initialiser les indices.
    splitNode(Root.get(), Data, Labels, Indices, 0); // Démarrer la division.
}

    /**
     * @brief Fonction pour diviser un nœud.
     */
void DecisionTreeSingle::splitNode(Tree* Node, const std::vector<std::vector<double>>& Data, const std::vector<double>& Labels, const std::vector<int>& Indices, int Depth) {
    double CurrentMSE = calculateMSE(Labels, Indices); // Calculer la MSE (Erreur Quadratique Moyenne) du nœud actuel.

    // Vérifier les conditions d'arrêt.
    if (Depth >= MaxDepth || Indices.size() < static_cast<size_t>(MinLeafLarge) || CurrentMSE < 1e-6) {
        Node->IsLeaf = true; // Marquer le nœud comme une feuille.
        Node->Prediction = calculateMean(Labels, Indices); // Calculer la moyenne pour la prédiction.
        return;
    }

    // Recherche du meilleur point de division.
    // BestFeature : Indice de la meilleure caractéristique.
    // BestThreshold : Seuil optimal pour la division.
    // BestImpurityDecrease : Réduction de l'impureté.
    auto [BestFeature, BestThreshold, BestImpurityDecrease] = findBestSplit(Data, Labels, Indices, CurrentMSE);
    
    // Pruning (élagage) précoce.
    // Si la réduction d'impureté est insuffisante, arrêter la division.
    if (BestFeature == -1 || BestImpurityDecrease < MinError) {
        Node->IsLeaf = true;
        Node->Prediction = calculateMean(Labels, Indices);
        return;
    }

    Node->FeatureIndex = BestFeature; // Stocker l'indice de la meilleure caractéristique.
    Node->MaxValue = BestThreshold; // Stocker le seuil optimal.

    // Diviser les données.
    std::vector<int> LeftIndices, RightIndices; // Créer les indices pour les feuilles gauche et droite.
    for (int Idx : Indices) {
        if (Data[Idx][BestFeature] <= BestThreshold) {
            LeftIndices.push_back(Idx);
        } else {
            RightIndices.push_back(Idx);
        }
    }

    Node->Left = std::make_unique<Tree>(); // Créer un nœud à gauche.
    Node->Right = std::make_unique<Tree>(); // Créer un nœud à droite.
    splitNode(Node->Left.get(), Data, Labels, LeftIndices, Depth + 1); // Appel récursif pour le nœud gauche.
    splitNode(Node->Right.get(), Data, Labels, RightIndices, Depth + 1); // Appel récursif pour le nœud droit.
}

/**
 * @brief Recherche du meilleur point de division.
 */
std::tuple<int, double, double> DecisionTreeSingle::findBestSplit(const std::vector<std::vector<double>>& Data, const std::vector<double>& Labels, const std::vector<int>& Indices, double CurrentMSE) {
    int BestFeature = -1; // Initialiser l'indice de la meilleure caractéristique à -1.
    double BestThreshold = 0.0; // Initialiser le seuil optimal.
    double BestImpurityDecrease = 0.0; // Initialiser la meilleure réduction d'impureté.

    size_t NumFeatures = Data[0].size(); // Obtenir le nombre de caractéristiques.
    auto SortedFeatureIndices = preSortFeatures(Data, Indices); // Pré-trier les caractéristiques.

    // Boucler sur chaque caractéristique.
    for (size_t Feature = 0; Feature < NumFeatures; ++Feature) {
        const auto& FeatureIndices = SortedFeatureIndices[Feature]; // Utiliser l'ordre trié.

        double LeftSum = 0.0, LeftSqSum = 0.0; // Initialiser les sommes pour le sous-ensemble gauche.
        size_t LeftCount = 0; // Compter les échantillons à gauche.

        double RightSum = 0.0, RightSqSum = 0.0;
        size_t RightCount = Indices.size();
        for (int Idx : FeatureIndices) {
            double Label = Labels[Idx];
            RightSum += Label; // Initialiser pour le sous-ensemble droit.
            RightSqSum += Label * Label; // Somme des carrés.
        }

        // Déplacer les échantillons du sous-ensemble droit vers le sous-ensemble gauche.
        for (size_t i = 0; i < FeatureIndices.size() - 1; ++i) {
            int Idx = FeatureIndices[i];
            double Value = Data[Idx][Feature];
            double Label = Labels[Idx];

            LeftSum += Label;
            LeftSqSum += Label * Label;
            LeftCount++;

            RightSum -= Label;
            RightSqSum -= Label * Label;
            RightCount--;

            double NextValue = Data[FeatureIndices[i + 1]][Feature];
            if (Value == NextValue) continue; // Ignorer les valeurs identiques.

            double LeftMean = LeftSum / LeftCount;
            double LeftMSE = (LeftSqSum - 2 * LeftMean * LeftSum + LeftCount * LeftMean * LeftMean) / LeftCount;

            double RightMean = RightSum / RightCount;
            double RightMSE = (RightSqSum - 2 * RightMean * RightSum + RightCount * RightMean * RightMean) / RightCount;

            double WeightedImpurity = (LeftMSE * LeftCount + RightMSE * RightCount) / Indices.size();
            double ImpurityDecrease = CurrentMSE - WeightedImpurity;

            if (ImpurityDecrease > BestImpurityDecrease) {
                BestImpurityDecrease = ImpurityDecrease;
                BestFeature = Feature;
                BestThreshold = (Value + NextValue) / 2.0;
            }
        }
    }
    return {BestFeature, BestThreshold, BestImpurityDecrease};
}

// Prédire pour un seul échantillon.
double DecisionTreeSingle::predict(const std::vector<double>& Sample) const {
    const Tree* CurrentNode = Root.get();
    while (!CurrentNode->IsLeaf) {
        if (Sample[CurrentNode->FeatureIndex] <= CurrentNode->MaxValue) {
            CurrentNode = CurrentNode->Left.get();
        } else {
            CurrentNode = CurrentNode->Right.get();
        }
    }
    return CurrentNode->Prediction;
}

// Calculer la moyenne des étiquettes.
double DecisionTreeSingle::calculateMean(const std::vector<double>& Labels, const std::vector<int>& Indices) {
    double Sum = 0.0;
    for (int Idx : Indices) Sum += Labels[Idx];
    return Sum / Indices.size();
}

// Calculer l'erreur quadratique moyenne (MSE).
double DecisionTreeSingle::calculateMSE(const std::vector<double>& Labels, const std::vector<int>& Indices) {
    double Mean = calculateMean(Labels, Indices);
    double MSE = 0.0;
    for (int Idx : Indices) {
        double Value = Labels[Idx];
        MSE += (Value - Mean) * (Value - Mean);
    }
    return MSE / Indices.size();
}

// Pré-trier les indices des caractéristiques.
std::vector<std::vector<int>> DecisionTreeSingle::preSortFeatures(const std::vector<std::vector<double>>& Data, const std::vector<int>& Indices) {
    size_t NumFeatures = Data[0].size();
    std::vector<std::vector<int>> SortedIndices(NumFeatures, Indices);

    for (size_t Feature = 0; Feature < NumFeatures; ++Feature) {
        std::sort(SortedIndices[Feature].begin(), SortedIndices[Feature].end(),
                  [&Data, Feature](int A, int B) {
                      return Data[A][Feature] < Data[B][Feature];
                  });
    }
    return SortedIndices;
}


void DecisionTreeSingle::serializeNode(const Tree* node, std::ostream& out) {
    if (!node) {
        out << "#\n"; // Use "#" to mark a null node
        return;
    }

    // Write the current node's data
    out << node->FeatureIndex << " "
        << node->MaxValue << " "
        << node->Prediction << " "
        << node->IsLeaf << "\n";

    // Recursively serialize the left and right children
    serializeNode(node->Left.get(), out);
    serializeNode(node->Right.get(), out);
}


void DecisionTreeSingle::saveTree(const std::string& filename) {
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path savePath = currentPath.parent_path() / "saved_models";

    // Create the directory if it doesn't exist
    std::filesystem::create_directories(savePath);
    
    std::ofstream outFile(  savePath / filename);
 
    if (!outFile.is_open()) {
        throw std::runtime_error("Unable to open file for saving the tree.");
    }

    // Save parameters
    outFile << MaxDepth << " " << MinLeafLarge << " " << MinError << "\n";

    // Serialize the tree structure
    serializeNode(Root.get(), outFile);

    outFile.close();
}

std::unique_ptr<DecisionTreeSingle::Tree> DecisionTreeSingle::deserializeNode(std::istream& in) {
    std::string line;
    if (!std::getline(in, line) || line == "#") {
        return nullptr; // Return null for "#" marker or end of input
    }

    auto node = std::make_unique<Tree>();
    std::istringstream ss(line);

    // Read node data
    ss >> node->FeatureIndex >> node->MaxValue >> node->Prediction >> node->IsLeaf;

    // Recursively deserialize the left and right children
    node->Left = deserializeNode(in);
    node->Right = deserializeNode(in);

    return node;
}

void DecisionTreeSingle::loadTree(const std::string& filename) {
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path savePath = currentPath.parent_path() / "saved_models";

    // Create the directory if it doesn't exist
    

    std::ifstream inFile(savePath / filename);


    if (!inFile.is_open()) {
        throw std::runtime_error("Unable to open file for loading the tree.");
    }

    // Load parameters
    inFile >> MaxDepth >> MinLeafLarge >> MinError;
    inFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the next line

    // Deserialize the tree structure
    Root = deserializeNode(inFile);

    inFile.close();
}



