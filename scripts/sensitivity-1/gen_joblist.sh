#!/usr/bin/env bash

# Configuration
PARAMS=( "w" "inv_t" "a_mho" )
SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE}")")" && pwd)
OUTPUT_FILE="${SCRIPT_DIR}/joblist.txt"
NSCENES=10
NTHREADS=4
TEMPLATE="./env.d/run.sh julia --threads=${NTHREADS}\
 ${SCRIPT_DIR}/run_model.jl"

# Clear the file (or create it if it doesn't exist)
: > "$OUTPUT_FILE"

# Loop and write lines

for param in "${PARAMS[@]}"
do
    for i in $(seq 1 $NSCENES)
    do
        echo "${TEMPLATE} ${param} ${i}" >> "$OUTPUT_FILE"
    done
done
