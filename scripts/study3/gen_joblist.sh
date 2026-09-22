#!/usr/bin/env sh

# Configuration
MODELS=( "mo" "ja" "ta" "fr" )
OUTPUT_FILE="scripts/study3/joblist.txt"
NSCENES=10
TEMPLATE="./env.d/run.sh julia scripts/study3/run_model.jl"

# Clear the file (or create it if it doesn't exist)
: > "$OUTPUT_FILE"

# Loop and write lines
for model in "${MODELS[@]}"
do
    for i in $(seq 1 $NSCENES)
    do
        echo "${TEMPLATE} ${model} ${i}" >> "$OUTPUT_FILE"
    done
done
