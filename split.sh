#!/bin/bash

INPUT="$1"
CHUNK=50000

if [ ! -f "$INPUT" ]; then
    echo "File not found: $INPUT"
    exit 1
fi

NAME="${INPUT%.*}"
EXT="${INPUT##*.}"

HEIGHT=$(ffprobe -v error \
-select_streams v:0 \
-show_entries stream=height \
-of csv=p=0 \
"$INPUT")

OFFSET=0
INDEX=1

while (( OFFSET < HEIGHT )); do
    CURRENT=$(( HEIGHT - OFFSET ))

    if (( CURRENT > CHUNK )); then
        CURRENT=$CHUNK
    fi

    ffmpeg -y -i "$INPUT" \
    -vf "crop=iw:${CURRENT}:0:${OFFSET}" \
    "${NAME}-${INDEX}.${EXT}"

    OFFSET=$(( OFFSET + CURRENT ))
    INDEX=$(( INDEX + 1 ))
done
