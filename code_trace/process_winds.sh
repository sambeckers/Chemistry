#!/bin/bash

# Don't exit on errors - we want to continue even if phantomanalysis crashes
set +e

# Script to run phantomanalysis on wind files in pairs from START_NUM onwards
# Processes files in pairs because analysis_trace.f90 needs two consecutive files:
# - First file: initialization only (reads particle IDs, sets up arrays)
# - Second file: actual trace file writing begins
# This prevents memory overflow while ensuring trace files are created

WIND_DIR="/net/vdesk/data2/beckers/MRP/v20_a25"
START_NUM=0

# Find the last wind file number
LAST_FILE=$(ls ${WIND_DIR}/wind_* | grep -E 'wind_[0-9]+$' | sort -V | tail -1)
LAST_NUM=$(basename ${LAST_FILE} | sed 's/wind_//')
LAST_NUM=$((10#${LAST_NUM}))
# LAST_NUM=1200

echo "Processing wind files from ${START_NUM} to ${LAST_NUM} in consecutive pairs"
echo "Start time: $(date)"

# Calculate total number of pairs
TOTAL_PAIRS=$((LAST_NUM - START_NUM))
CURRENT_PAIR=0

# Track failed pairs
FAILED_PAIRS=()

# Loop through files, processing consecutive pairs
for i in $(seq ${START_NUM} $((LAST_NUM - 1))); do
    # Increment counter and show progress
    CURRENT_PAIR=$((CURRENT_PAIR + 1))
    PERCENT=$((CURRENT_PAIR * 100 / TOTAL_PAIRS))
    
    # Format the file numbers with leading zeros (5 digits)
    FILE_NUM1=$(printf "%05d" $i)
    FILE_NUM2=$(printf "%05d" $((i + 1)))
    WIND_FILE1="${WIND_DIR}/wind_${FILE_NUM1}"
    WIND_FILE2="${WIND_DIR}/wind_${FILE_NUM2}"
    
    # Check if both files exist
    if [ -f "${WIND_FILE1}" ] && [ -f "${WIND_FILE2}" ]; then
        echo "========================================="
        echo "Progress: [${CURRENT_PAIR}/${TOTAL_PAIRS}] (${PERCENT}%)"
        echo "Processing pair: wind_${FILE_NUM1} and wind_${FILE_NUM2} ($(date))"
        echo "========================================="
        
        # Run phantomanalysis on consecutive pair and capture output
        # Use || true to prevent script from exiting on segfault
        OUTPUT=$(./phantomanalysis ${WIND_FILE1} ${WIND_FILE2} 2>&1 || true)
        EXIT_CODE=$?
        
        # Print the output
        echo "$OUTPUT"
        
        # Check if success message appears in output
        if echo "$OUTPUT" | grep -q "may your paper be a happy one"; then
            echo "Successfully processed wind_${FILE_NUM1} -> wind_${FILE_NUM2}"
        else
            echo "ERROR: Failed processing wind_${FILE_NUM1} -> wind_${FILE_NUM2} (no success message, continuing...)"
            FAILED_PAIRS+=("${FILE_NUM1}-${FILE_NUM2}")
        fi
    else
        echo "Warning: One or both files (wind_${FILE_NUM1}, wind_${FILE_NUM2}) do not exist, skipping..."
    fi
done

echo "========================================="
echo "All file pairs processed!"
echo "End time: $(date)"

# Report failed pairs
if [ ${#FAILED_PAIRS[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: ${#FAILED_PAIRS[@]} pair(s) failed:"
    for pair in "${FAILED_PAIRS[@]}"; do
        echo "  - wind_${pair}"
    done
fi
echo "========================================="
echo "End time: $(date)"
echo "========================================="
