#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ "$#" -ge 1 ]; then
    cmake --build build -j --target $1
else
    cmake --build build -j
fi
