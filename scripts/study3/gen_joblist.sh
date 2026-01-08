#!/usr/bin/env sh

# Configuration
OUTPUT_FILE="scripts/load/joblist.txt"
NSCENES=10
NTHREADS=8
TEMPLATE="./env.d/run.sh julia --threads=${NTHREADS}\
 scripts/load/run_model.jl"

# Clear the file (or create it if it doesn't exist)
: > "$OUTPUT_FILE"

# Loop and write lines
for i in $(seq 1 $NSCENES)
do
    echo "${TEMPLATE} mo ${i}" >> "$OUTPUT_FILE"
done
for i in $(seq 1 $NSCENES)
do
    echo "${TEMPLATE} ja ${i}" >> "$OUTPUT_FILE"
done
for i in $(seq 1 $NSCENES)
do
    echo "${TEMPLATE} ta ${i}" >> "$OUTPUT_FILE"
done
for i in $(seq 1 $NSCENES)
do
    echo "${TEMPLATE} fr ${i}" >> "$OUTPUT_FILE"
done
