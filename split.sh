#!/usr/bin/env bash
# split.sh — split a very long PNG into equal halves vertically.
#
# Usage:
#   bash split.sh input.png output_top.png output_bottom.png
#
# Requires: ImageMagick (convert, identify).
# Used when a chapter is too large for some downstream tools and must be
# processed in two halves, then re-merged with merge.sh.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 input.png top_output.png bottom_output.png"
  exit 1
fi

INPUT="$1"
TOP_OUT="$2"
BOTTOM_OUT="$3"

HEIGHT=$(identify -format "%h" "$INPUT")
HALF=$(( HEIGHT / 2 ))

convert "$INPUT" -crop "x${HALF}+0+0" +repage "$TOP_OUT"
convert "$INPUT" -crop "x${HALF}+0+${HALF}" +repage "$BOTTOM_OUT"

echo "Split at row $HALF: $TOP_OUT + $BOTTOM_OUT"
