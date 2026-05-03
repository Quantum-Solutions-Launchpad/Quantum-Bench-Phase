#!/usr/bin/env bash
set -euo pipefail

quaph run analytic haldane \
    --n-sites 6 \
    --log-dir logs \
    --plot-dir plots
