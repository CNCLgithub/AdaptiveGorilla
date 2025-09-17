#!/usr/bin/env sh

# Configuration
MODEL="$1"
OUTPUT_FILE="scripts/most/${MODEL}-joblist.txt"
NSCENES=6
NTHREADS=8
TEMPLATE="./env.d/run.sh julia --threads=${NTHREADS}\
 scripts/most/run_model.jl"

# Clear the file (or create it if it doesn't exist)
: > "$OUTPUT_FILE"

# Loop and write lines
for i in $(seq 1 $NSCENES)
do
    echo "${TEMPLATE} ${MODEL} ${i}" >> "$OUTPUT_FILE"
done
