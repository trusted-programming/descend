#!/bin/bash
# Script to update MLIR dialect include paths for the current environment
# This is needed because melior::dialect! macro only accepts string literals
# and cannot use env!() or concat!() for path resolution.

set -e

# Get the project root directory (where this script is called from)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INCLUDE_PATH="${PROJECT_ROOT}/AscendNPU-IR/bishengir/include"
DIALECTS_FILE="${PROJECT_ROOT}/src/codegen/mlir/dialects.rs"

echo "Updating dialect paths in: $DIALECTS_FILE"
echo "Include path will be set to: $INCLUDE_PATH"

# Use sed to replace the include_directories paths
# The pattern matches: include_directories: ["ANY_PATH"],
# and replaces it with the correct path
sed -i.bak "s|include_directories: \[\"[^\"]*\"\]|include_directories: [\"${INCLUDE_PATH}\"]|g" "$DIALECTS_FILE"

echo "✓ Paths updated successfully!"
echo "  Backup saved to: ${DIALECTS_FILE}.bak"
echo ""
echo "You can now build the project with: cargo build"

