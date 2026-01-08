#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <source_cluster> <dest_cluster> [results_path]"
    echo "Example: $0 vulcan mila"
    echo "         $0 vulcan mila scratch/hydra-runs/finetune-with-strategy/final-results.jsonl"
    exit 1
fi

SOURCE_CLUSTER=$1
DEST_CLUSTER=$2
RESULTS_PATH=${3:-scratch/hydra-runs/finetune-with-strategy/final-results.jsonl}

SOURCE_CLUSTER_NAME=$SOURCE_CLUSTER

DEST_FILE="final-results-${SOURCE_CLUSTER_NAME}.jsonl"
DEST_DIR=$(dirname $RESULTS_PATH)

echo "Syncing results from ${SOURCE_CLUSTER}:${RESULTS_PATH}"
echo "To ${DEST_CLUSTER}:${DEST_DIR}/${DEST_FILE}"
echo ""

scp -3 ${SOURCE_CLUSTER}:${RESULTS_PATH} ${DEST_CLUSTER}:${DEST_DIR}/${DEST_FILE}

if [ $? -ne 0 ]; then
    echo "Error syncing results file"
    exit 1
fi

GENERATIONS_DIR=$(dirname $RESULTS_PATH)/generations
DEST_GENERATIONS_DIR="${DEST_DIR}/generations-${SOURCE_CLUSTER_NAME}"

echo ""
echo "Syncing generations from ${SOURCE_CLUSTER}:${GENERATIONS_DIR}"
echo "To ${DEST_CLUSTER}:${DEST_GENERATIONS_DIR}"
echo ""

scp -3 -r ${SOURCE_CLUSTER}:${GENERATIONS_DIR} ${DEST_CLUSTER}:${DEST_GENERATIONS_DIR}

if [ $? -ne 0 ]; then
    echo "Error syncing generations directory"
    exit 1
fi

echo ""
echo "Successfully synced:"
echo "  - Results to ${DEST_FILE}"
echo "  - Generations to ${DEST_GENERATIONS_DIR}"
echo ""
echo "Run the merge script to combine results:"
echo "  python scripts/merge_results.py ${DEST_DIR}/${DEST_FILE} ${DEST_DIR}/final-results.jsonl"
