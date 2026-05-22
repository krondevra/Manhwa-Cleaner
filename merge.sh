#!/usr/bin/env bash
# merge.sh — merge two PNG images vertically.
#
# Usage:
#   bash merge.sh top.png bottom.png output.png
#
# Requires: ImageMagick (convert).
# Used when a chapter is exported as two separate halves and must be joined
# before the ML cleaner processes it.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 top.png bottom.png output.png"
  exit 1
fi

TOP="$1"
BOTTOM="$2"
OUTPUT="$3"

convert "$TOP" "$BOTTOM" -append "$OUTPUT"
echo "Merged: $OUTPUT"
