#!/bin/bash

# Script to calculate column densities for all wind dumps
# Processes each dump individually to avoid memory issues

# Configuration
MAP="v10_a09"  # Change this to v10_a09 or other map as needed
START_DUMP=0  # Starting dump number (set to 0 to process all)
END_DUMP=9999    # Ending dump number (set to 9999 for all available)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
DUMP_DIR="${PARENT_DIR}/${MAP}"

echo "========================================="
echo "Column Density Calculation"
echo "========================================="
echo "Map: ${MAP}"
echo "Dump directory: ${DUMP_DIR}"
echo "Start time: $(date)"
echo ""

# Check if dump directory exists
if [ ! -d "${DUMP_DIR}" ]; then
    echo "ERROR: Dump directory ${DUMP_DIR} does not exist!"
    exit 1
fi

# Find all wind dump files and extract numbers
DUMP_FILES=$(ls ${DUMP_DIR}/wind_* 2>/dev/null | grep -E 'wind_[0-9]{5}$' | sort -V)

if [ -z "$DUMP_FILES" ]; then
    echo "ERROR: No dump files found in ${DUMP_DIR}"
    exit 1
fi

# Count total dumps
TOTAL_DUMPS=$(echo "$DUMP_FILES" | wc -l)
echo "Found ${TOTAL_DUMPS} dump files"
echo ""

# Process each dump
PROCESSED=0
SKIPPED=0
FAILED=0

for DUMP_FILE in $DUMP_FILES; do
    # Extract dump number
    DUMP_NUM=$(basename ${DUMP_FILE} | sed 's/wind_//' | sed 's/^0*//')
    
    # Check if within range
    if [ ${DUMP_NUM} -lt ${START_DUMP} ] || [ ${DUMP_NUM} -gt ${END_DUMP} ]; then
        continue
    fi
    
    # Check if output already exists
    OUTPUT_FILE="${SCRIPT_DIR}/PhotoData/AV_${DUMP_NUM}_nrays192_nside8.txt"
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "✓ Dump ${DUMP_NUM} already processed, skipping..."
        ((SKIPPED++))
        continue
    fi
    
    echo "========================================="
    echo "Processing dump ${DUMP_NUM} ($(date))"
    echo "========================================="
    
    # Modify the Python script to use this dump number
    # We'll pass it as an environment variable
    export DUMP_NUM=${DUMP_NUM}
    export MAP_NAME=${MAP}
    
    # Run the Python script
    cd ${SCRIPT_DIR}
    python column_densities.py
    
    # Check if successful
    if [ $? -eq 0 ] && [ -f "${OUTPUT_FILE}" ]; then
        echo "Successfully processed dump ${DUMP_NUM}"
        ((PROCESSED++))
    else
        echo "Failed to process dump ${DUMP_NUM}"
        ((FAILED++))
        # Uncomment to stop on first error
        # exit 1
    fi
    
    echo ""
done

echo "========================================="
echo "Summary"
echo "========================================="
echo "Total dumps found: ${TOTAL_DUMPS}"
echo "Processed: ${PROCESSED}"
echo "Skipped (already done): ${SKIPPED}"
echo "Failed: ${FAILED}"
echo "End time: $(date)"
echo "========================================="
