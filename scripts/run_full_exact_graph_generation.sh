#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT="$HOME/gnn_thesis"
DATA_ROOT="$PROJECT/data/network-flows-merged-net101x"
OUTPUT_ROOT="$PROJECT/unraveled_graphs_150000_merged_captures_exact_57_72"
GENERATOR="$PROJECT/unraveled_graph_representation.py"

PYTHON_BIN="$(command -v python)"

echo "======================================================================"
echo "GENERAZIONE COMPLETA TDG/e-TDG"
echo "Data di avvio: $(date)"
echo "Python: $PYTHON_BIN"
echo "Input: $DATA_ROOT"
echo "Output: $OUTPUT_ROOT"
echo "Generatore: $GENERATOR"
echo "Modalità e-TDG: exact"
echo "Snapshot interval: 150000 ms"
echo "======================================================================"

if [[ ! -f "$GENERATOR" ]]; then
    echo "ERRORE: generatore non trovato: $GENERATOR"
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERRORE: cartella input non trovata: $DATA_ROOT"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

mapfile -t CSV_FILES < <(
    find "$DATA_ROOT" \
        -mindepth 2 \
        -maxdepth 2 \
        -type f \
        -name "merged_net101x_Flow_labeled.csv" \
        | sort
)

TOTAL_FILES="${#CSV_FILES[@]}"

if [[ "$TOTAL_FILES" -eq 0 ]]; then
    echo "ERRORE: nessun CSV merged trovato"
    exit 1
fi

echo "CSV individuati: $TOTAL_FILES"
printf '%s\n' "${CSV_FILES[@]}"
echo

CURRENT=0

for CSV_FILE in "${CSV_FILES[@]}"; do
    CURRENT=$((CURRENT + 1))

    INPUT_DIR="$(dirname "$CSV_FILE")"
    WEEK_FOLDER="$(basename "$INPUT_DIR")"
    OUTPUT_WEEK="$OUTPUT_ROOT/$WEEK_FOLDER"

    mkdir -p "$OUTPUT_WEEK"

    echo
    echo "======================================================================"
    echo "[$CURRENT/$TOTAL_FILES] $WEEK_FOLDER"
    echo "Inizio: $(date)"
    echo "Input: $INPUT_DIR"
    echo "Output: $OUTPUT_WEEK"
    echo "======================================================================"

    "$PYTHON_BIN" -u "$GENERATOR" \
        --data_path "$INPUT_DIR" \
        --out_path "$OUTPUT_WEEK" \
        --graph_repre ALL \
        --snapshot_interval 150000 \
        --etdg_mode exact

    echo "Completata: $WEEK_FOLDER"
    echo "Fine: $(date)"

    TDG_COUNT=$(find "$OUTPUT_ROOT" -type f -path "*/tdg_graph/*/*.pkl" | wc -l)
    ETDG_COUNT=$(find "$OUTPUT_ROOT" -type f -path "*/etdg_graph/*/*.pkl" | wc -l)

    echo "TDG salvati finora: $TDG_COUNT"
    echo "e-TDG salvati finora: $ETDG_COUNT"
    du -sh "$OUTPUT_ROOT"
done

echo
echo "======================================================================"
echo "GENERAZIONE COMPLETATA"
echo "Data di conclusione: $(date)"
echo "======================================================================"

TDG_COUNT=$(find "$OUTPUT_ROOT" -type f -path "*/tdg_graph/*/*.pkl" | wc -l)
ETDG_COUNT=$(find "$OUTPUT_ROOT" -type f -path "*/etdg_graph/*/*.pkl" | wc -l)

echo "TDG finali: $TDG_COUNT"
echo "e-TDG finali: $ETDG_COUNT"
echo "Totale file grafi: $((TDG_COUNT + ETDG_COUNT))"

du -sh "$OUTPUT_ROOT"
