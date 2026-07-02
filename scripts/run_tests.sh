#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if python3 - <<'PY' >/dev/null 2>&1
import pytest
PY
then
  exec python3 -m pytest "$@"
fi
# Fallback for this lightweight repo when pytest is not installed. Supports test file paths.
if [ "$#" -eq 0 ]; then
  exec python3 -m unittest discover -s tests -p 'test*.py'
fi
mods=()
args=()
for arg in "$@"; do
  case "$arg" in
    -*) args+=("$arg") ;;
    *.py) mods+=("${arg%.py}") ;;
    *) mods+=("$arg") ;;
  esac
done
if [ "${#mods[@]}" -eq 0 ]; then
  exec python3 -m unittest discover -s tests -p 'test*.py' "${args[@]}"
fi
for i in "${!mods[@]}"; do
  mods[$i]="${mods[$i]//\//.}"
done
exec python3 -m unittest "${mods[@]}"
