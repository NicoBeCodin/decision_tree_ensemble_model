#!/bin/bash

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher un message d'aide
show_help() {
    echo -e "${BLUE}Script de compilation pour DecisionTreeEnsembleModel${NC}"
    echo ""
    echo "Usage: ./build.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help       Affiche ce message d'aide"
    echo "  -p, --parallel   Compile avec OpenMP (parallèle)"
    echo "  -s, --sequential Compile sans OpenMP (séquentiel)"
    echo "  -c, --clean      Nettoie le dossier build avant compilation"
    echo ""
    echo "Si aucune option n'est spécifiée, le script demandera interactivement."
}

# Fonction pour nettoyer le dossier build
clean_build() {
    echo -e "${YELLOW}Nettoyage du dossier build...${NC}"
    rm -rf build
    mkdir -p build
}

# Fonction pour compiler avec OpenMP
build_with_openmp() {
    echo -e "${GREEN}Compilation avec OpenMP (parallèle)...${NC}"
    mkdir -p build
    cd build
    cmake -DUSE_OPENMP=ON ..
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
    cd ..
    echo -e "${GREEN}Compilation terminée.${NC}"
}

# Fonction pour compiler sans OpenMP
build_without_openmp() {
    echo -e "${YELLOW}Compilation sans OpenMP (séquentiel)...${NC}"
    mkdir -p build
    cd build
    cmake -DUSE_OPENMP=OFF ..
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
    cd ..
    echo -e "${GREEN}Compilation terminée.${NC}"
}

# Traitement des arguments
if [ $# -gt 0 ]; then
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -p|--parallel)
                USE_OPENMP=1
                ;;
            -s|--sequential)
                USE_OPENMP=0
                ;;
            -c|--clean)
                CLEAN_BUILD=1
                ;;
            *)
                echo -e "${RED}Option inconnue: $1${NC}"
                show_help
                exit 1
                ;;
        esac
        shift
    done
else
    # Mode interactif si aucun argument n'est fourni
    echo -e "${BLUE}Configuration de la compilation${NC}"
    echo ""
    echo -e "Voulez-vous utiliser OpenMP pour la compilation parallèle ?"
    echo -e "1) ${GREEN}Oui${NC} - Compilation avec OpenMP (parallèle)"
    echo -e "2) ${YELLOW}Non${NC} - Compilation sans OpenMP (séquentiel)"
    echo -e "3) ${RED}Annuler${NC} - Quitter sans compiler"
    echo ""
    read -p "Votre choix [1-3]: " choice
    
    case $choice in
        1)
            USE_OPENMP=1
            ;;
        2)
            USE_OPENMP=0
            ;;
        3)
            echo -e "${RED}Compilation annulée.${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Choix invalide. Compilation annulée.${NC}"
            exit 1
            ;;
    esac
    
    echo ""
    echo -e "Voulez-vous nettoyer le dossier build avant la compilation ?"
    echo -e "1) ${GREEN}Oui${NC} - Nettoyer le dossier build"
    echo -e "2) ${YELLOW}Non${NC} - Conserver le dossier build existant"
    echo ""
    read -p "Votre choix [1-2]: " clean_choice
    
    case $clean_choice in
        1)
            CLEAN_BUILD=1
            ;;
        2)
            CLEAN_BUILD=0
            ;;
        *)
            echo -e "${RED}Choix invalide. Le dossier build ne sera pas nettoyé.${NC}"
            CLEAN_BUILD=0
            ;;
    esac
fi

# Exécution des actions
if [ "$CLEAN_BUILD" = "1" ]; then
    clean_build
fi

if [ "$USE_OPENMP" = "1" ]; then
    build_with_openmp
elif [ "$USE_OPENMP" = "0" ]; then
    build_without_openmp
fi

echo -e "${BLUE}Terminé !${NC}" 