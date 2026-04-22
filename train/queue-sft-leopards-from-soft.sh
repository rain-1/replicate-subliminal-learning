#!/bin/bash
# Wait for an existing run to finish, then launch the SFT-from-soft control.

set -euo pipefail

WAIT_FOR_PID="${WAIT_FOR_PID:-}"

if [ -n "$WAIT_FOR_PID" ]; then
    echo "[sft-from-soft] waiting for PID $WAIT_FOR_PID"
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
        sleep 60
    done
fi

echo "[sft-from-soft] starting"
bash train/launch-sft-leopards-from-soft.sh
