#!/usr/bin/env bash

# Configuration
MODELS=( "mo" "ja" "ta" "fr" )
OUTPUT_FILE="scripts/study2/joblist.txt"
NSCENES=6
NTHREADS=8
TEMPLATE="./env.d/run.sh julia --threads=${NTHREADS}\
 scripts/study2/run_model.jl"

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
