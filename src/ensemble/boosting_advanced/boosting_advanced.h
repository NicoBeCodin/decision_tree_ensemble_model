#ifndef BOOSTING_ADVANCED_H
#define BOOSTING_ADVANCED_H

#include <vector>
#include "boosting_improved.h"

// Classe d'interface pour la compatibilité avec le code existant
class AdvancedGBDT {
public:
    // Constructeur compatible avec le code existant
    AdvancedGBDT(int n_estimators, int max_depth, double learning_rate = 0.1,
                 bool useDart = false, double drop_rate = 0.1, double skip_rate = 0.0)
        : impl(n_estimators, max_depth, learning_rate, useDart, drop_rate, skip_rate,
               ImprovedGBDT::BinningMethod::NONE, 0) {}

    // Méthode d'entraînement
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        impl.fit(X, y);
    }

    // Prédiction pour plusieurs échantillons
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const {
        return impl.predict(X);
    }

    // Prédiction pour un seul échantillon
    double predict(const std::vector<double>& x) const {
        return impl.predict(x);
    }

private:
    ImprovedGBDT impl; // Utilisation de l'implémentation améliorée
};

#endif // BOOSTING_ADVANCED_H