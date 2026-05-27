#!/bin/bash

PREFIX="$1"

if [ -z "$PREFIX" ]; then
    echo "Usage: ./merge.sh <prefix>"
    exit 1
fi

OUTPUT="${PREFIX}_cleaned.png"

FILES=($(ls ${PREFIX}-*_cleaned.png 2>/dev/null | sort -V))

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No files found"
    exit 1
fi

CMD="ffmpeg -y"

FILTER=""

for i in "${!FILES[@]}"; do
    CMD="$CMD -i \"${FILES[$i]}\""
    FILTER="${FILTER}[${i}:v]"
done

FILTER="${FILTER}vstack=inputs=${#FILES[@]}[out]"

CMD="$CMD -filter_complex \"$FILTER\" -map \"[out]\" \"$OUTPUT\""

eval $CMD

echo "Saved: $OUTPUT"
