#!/usr/bin/env sh

# Configuration
OUTPUT_FILE="scripts/most/joblist.txt"
NSCENES=6
NTHREADS=8
TEMPLATE="./env.d/run.sh julia --threads=${NTHREADS}\
 scripts/most/run_model.jl"

# Clear the file (or create it if it doesn't exist)
: > "$OUTPUT_FILE"

# Loop and write lines
for i in $(seq 1 $NSCENES)
do
    echo "${TEMPLATE} MO ${i}" >> "$OUTPUT_FILE"
done
for i in $(seq 1 $NSCENES)
do
    echo "${TEMPLATE} AC ${i}" >> "$OUTPUT_FILE"
done
for i in $(seq 1 $NSCENES)
do
    echo "${TEMPLATE} FR ${i}" >> "$OUTPUT_FILE"
done
