#include "decision_tree_XGBoost.h"

/**
 * @brief Constructeur pour initialiser l'arbre de décision avec les paramètres spécifiés
 * @param MaxDepth Profondeur maximale de l'arbre
 * @param MinLeafSize Taille minimale d'une feuille
 * @param Lambda Paramètre de régularisation L2
 * @param Gamma Seuil de gain minimal pour une scission
 */
DecisionTreeXGBoost::DecisionTreeXGBoost(int MaxDepth, int MinLeafSize, double Lambda, double Gamma)
    : MaxDepth(MaxDepth), MinLeafSize(MinLeafSize), Lambda(Lambda), Gamma(Gamma), Root(nullptr) {}

/**
 * @brief Entraîner l'arbre de décision
 * @param Data Matrice des caractéristiques d'entraînement
 * @param Labels Vecteur des étiquettes cibles
 * @param Predictions Vecteur des prédictions actuelles (sera utilisé pour calculer les gradients)
 */
void DecisionTreeXGBoost::train(const std::vector<std::vector<double>>& Data, 
                                const std::vector<double>& Labels, 
                                std::vector<double>& Predictions) {
    Root = std::make_unique<Tree>();
    std::vector<int> Indices(Data.size());
    std::iota(Indices.begin(), Indices.end(), 0);

    // Calculer gradients et hessians
    std::vector<double> Gradients(Data.size()), Hessians(Data.size());
    computeGradientsAndHessians(Labels, Predictions, Gradients, Hessians);

    // Construire l'arbre
    splitNode(Root.get(), Data, Gradients, Hessians, Indices, 0);
}

/**
 * @brief Prédire la valeur pour un seul échantillon
 * @param Sample Vecteur représentant un échantillon unique
 * @return La prédiction faite par l'arbre pour cet échantillon
 */
double DecisionTreeXGBoost::predict(const std::vector<double>& Sample) const {
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

/**
 * @brief Calculer les gradients et hessians pour chaque échantillon
 * @param Labels Vecteur des étiquettes réelles
 * @param Predictions Vecteur des prédictions actuelles
 * @param Gradients Sortie contenant les gradients calculés
 * @param Hessians Sortie contenant les hessians calculés
 */
void DecisionTreeXGBoost::computeGradientsAndHessians(
    const std::vector<double>& Labels,
    const std::vector<double>& Predictions,
    std::vector<double>& Gradients,
    std::vector<double>& Hessians) {
    for (size_t i = 0; i < Labels.size(); ++i) {
        double pred = Predictions[i];
        double label = Labels[i];
        Gradients[i] = 2 * (pred - label); // Gradient de MSE
        Hessians[i] = 2.0;                // Hessian constant pour MSE
    }
}

/**
 * @brief Diviser un nœud dans l'arbre
 * @param Node Pointeur vers le nœud à diviser
 * @param Data Matrice des caractéristiques
 * @param Gradients Vecteur des gradients
 * @param Hessians Vecteur des hessians
 * @param Indices Indices des échantillons disponibles pour cette division
 * @param Depth Profondeur actuelle du nœud
 */
void DecisionTreeXGBoost::splitNode(Tree* Node, const std::vector<std::vector<double>>& Data, 
                                    const std::vector<double>& Gradients,
                                    const std::vector<double>& Hessians,
                                    const std::vector<int>& Indices, int Depth) {
    // Critère d'arrêt
    if (Depth >= MaxDepth || Indices.size() < static_cast<size_t>(MinLeafSize)) {
        Node->IsLeaf = true;
        Node->Prediction = calculateLeafWeight(Indices, Gradients, Hessians);
        return;
    }

    // Trouver la meilleure scission
    auto [BestFeature, BestThreshold, BestGain] = findBestSplit(Data, Gradients, Hessians, Indices);

    if (BestFeature == -1 || BestGain < Gamma) { // Pas de gain suffisant
        Node->IsLeaf = true;
        Node->Prediction = calculateLeafWeight(Indices, Gradients, Hessians);
        return;
    }

    Node->FeatureIndex = BestFeature;
    Node->MaxValue = BestThreshold;
    Node->GainImprovement = BestGain;

    // Partitionner les indices
    std::vector<int> LeftIndices, RightIndices;
    for (int idx : Indices) {
        if (Data[idx][BestFeature] <= BestThreshold) {
            LeftIndices.push_back(idx);
        } else {
            RightIndices.push_back(idx);
        }
    }

    Node->Left = std::make_unique<Tree>();
    Node->Right = std::make_unique<Tree>();
    splitNode(Node->Left.get(), Data, Gradients, Hessians, LeftIndices, Depth + 1);
    splitNode(Node->Right.get(), Data, Gradients, Hessians, RightIndices, Depth + 1);
}

/**
 * @brief Trouver la meilleure division possible pour un nœud
 * @param Data Matrice des caractéristiques
 * @param Gradients Vecteur des gradients
 * @param Hessians Vecteur des hessians
 * @param Indices Indices des échantillons disponibles pour cette division
 * @return Tuple contenant la meilleure caractéristique, le seuil et le gain
 */
std::tuple<int, double, double> DecisionTreeXGBoost::findBestSplit(
    const std::vector<std::vector<double>>& Data,
    const std::vector<double>& Gradients,
    const std::vector<double>& Hessians,
    const std::vector<int>& Indices) {
    
    int BestFeature = -1;
    double BestThreshold = 0.0;
    double BestGain = 0.0;

    double G = sumGradients(Gradients, Indices);
    double H = sumHessians(Hessians, Indices);
    double CurrentScore = G * G / (H + Lambda);

    for (size_t feature = 0; feature < Data[0].size(); ++feature) {
        std::vector<std::pair<double, int>> SortedValues;
        for (int idx : Indices) {
            SortedValues.emplace_back(Data[idx][feature], idx);
        }
        std::sort(SortedValues.begin(), SortedValues.end());

        double GL = 0.0, HL = 0.0;
        for (size_t i = 0; i < SortedValues.size() - 1; ++i) {
            GL += Gradients[SortedValues[i].second];
            HL += Hessians[SortedValues[i].second];
            double GR = G - GL;
            double HR = H - HL;

            if (HL >= MinLeafSize && HR >= MinLeafSize) {
                double Gain = GL * GL / (HL + Lambda) + 
                             GR * GR / (HR + Lambda) - 
                             CurrentScore - Gamma;

                if (Gain > BestGain && 
                    SortedValues[i].first != SortedValues[i + 1].first) {
                    BestGain = Gain;
                    BestFeature = feature;
                    BestThreshold = (SortedValues[i].first + 
                                   SortedValues[i + 1].first) / 2.0;
                }
            }
        }
    }

    return {BestFeature, BestThreshold, BestGain};
}

/**
 * @brief Calculer le poids d'une feuille
 * @param Indices Indices des échantillons dans la feuille
 * @param Gradients Vecteur des gradients
 * @param Hessians Vecteur des hessians
 * @return Le poids calculé pour cette feuille
 */
double DecisionTreeXGBoost::calculateLeafWeight(const std::vector<int>& Indices,
                                                const std::vector<double>& Gradients,
                                                const std::vector<double>& Hessians) {
    double G = sumGradients(Gradients, Indices);
    double H = sumHessians(Hessians, Indices);
    return -G / (H + Lambda);
}

/**
 * @brief Calculer la somme des gradients pour les échantillons spécifiés
 * @param Gradients Vecteur des gradients
 * @param Indices Indices des échantillons
 * @return La somme des gradients
 */
double DecisionTreeXGBoost::sumGradients(const std::vector<double>& Gradients, const std::vector<int>& Indices) {
    double Sum = 0.0;
    for (int idx : Indices) {
        Sum += Gradients[idx];
    }
    return Sum;
}

/**
 * @brief Calculer la somme des hessians pour les échantillons spécifiés
 * @param Hessians Vecteur des hessians
 * @param Indices Indices des échantillons
 * @return La somme des hessians
 */
double DecisionTreeXGBoost::sumHessians(const std::vector<double>& Hessians, const std::vector<int>& Indices) {
    double Sum = 0.0;
    for (int idx : Indices) {
        Sum += Hessians[idx];
    }
    return Sum;
}

/**
 * @brief Sauvegarder l'arbre dans un fichier
 * @param filename Nom du fichier où l'arbre sera sauvegardé
 */
void DecisionTreeXGBoost::saveTree(const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        throw std::runtime_error("Unable to open file for saving the tree.");
    }
    serializeNode(Root.get(), outFile);
    outFile.close();
}

/**
 * @brief Charger un arbre depuis un fichier
 * @param filename Nom du fichier contenant l'arbre sauvegardé
 */
void DecisionTreeXGBoost::loadTree(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        throw std::runtime_error("Unable to open file for loading the tree.");
    }
    Root = deserializeNode(inFile);
    inFile.close();
}

/**
 * @brief Sérialiser un nœud pour l'écriture dans un fichier
 * @param node Pointeur vers le nœud à sérialiser
 * @param out Flux de sortie pour écrire les données du nœud
 */
void DecisionTreeXGBoost::serializeNode(const Tree* node, std::ostream& out) {
    if (!node) {
        out << "#\n";
        return;
    }
    out << node->FeatureIndex << " " << node->MaxValue << " " << node->Prediction << " " << node->IsLeaf << "\n";
    serializeNode(node->Left.get(), out);
    serializeNode(node->Right.get(), out);
}

/**
 * @brief Désérialiser un nœud à partir d'un flux d'entrée
 * @param in Flux d'entrée contenant les données sérialisées
 * @return Un pointeur unique vers le nœud désérialisé
 */
std::unique_ptr<DecisionTreeXGBoost::Tree> DecisionTreeXGBoost::deserializeNode(std::istream& in) {
    std::string line;
    if (!std::getline(in, line) || line == "#") return nullptr;

    auto node = std::make_unique<Tree>();
    std::istringstream ss(line);
    ss >> node->FeatureIndex >> node->MaxValue >> node->Prediction >> node->IsLeaf;

    node->Left = deserializeNode(in);
    node->Right = deserializeNode(in);

    return node;
}

std::map<int, double> DecisionTreeXGBoost::getFeatureImportance() const {
    std::map<int, double> importance;
    calculateFeatureImportanceRecursive(Root.get(), importance);
    return importance;
}

void DecisionTreeXGBoost::calculateFeatureImportanceRecursive(
    const Tree* node, std::map<int, double>& importance) const {
    if (!node || node->IsLeaf) {
        return;
    }

    // Ajouter le gain d'amélioration à l'importance de la caractéristique
    importance[node->FeatureIndex] += node->GainImprovement;

    // Récursivement calculer l'importance pour les sous-arbres
    calculateFeatureImportanceRecursive(node->Left.get(), importance);
    calculateFeatureImportanceRecursive(node->Right.get(), importance);
}
