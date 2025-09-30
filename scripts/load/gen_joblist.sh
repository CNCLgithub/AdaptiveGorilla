#!/usr/bin/env sh

# Configuration
OUTPUT_FILE="scripts/load/joblist.txt"
NSCENES=9
NTHREADS=8
TEMPLATE="./env.d/run.sh julia --threads=${NTHREADS}\
 scripts/load/run_model.jl"

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
