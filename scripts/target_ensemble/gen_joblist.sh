#!/usr/bin/env sh

# Configuration
MODEL="full"
OUTPUT_FILE="scripts/target_ensemble/joblist.txt"
NSCENES=6
NTHREADS=8
TEMPLATE="./env.d/run.sh julia --threads=${NTHREADS}\
 scripts/target_ensemble/run_model.jl"

# Clear the file (or create it if it doesn't exist)
: > "$OUTPUT_FILE"

# Loop and write lines
for i in $(seq 1 $NSCENES)
do
    echo "${TEMPLATE} ${MODEL} ${i}" >> "$OUTPUT_FILE"
done
