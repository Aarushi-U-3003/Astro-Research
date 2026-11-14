#!/bin/bash
# Pulls the benchmark reports from GCS and prints one table.
#
#   ./benchmark/collect.sh <bucket-name>
set -euo pipefail

BUCKET="${1:?usage: collect.sh <bucket-name>}"
TMP=$(mktemp -d)

gcloud storage cp "gs://${BUCKET}/benchmark/results/timing_*.txt" "${TMP}/" 2>/dev/null \
  || { echo "No reports yet in gs://${BUCKET}/benchmark/results/"; exit 1; }

printf '%-4s %-8s %-7s %-6s %-10s %-9s %-8s\n' \
  IDX DL THETA EXIT RUN_HOURS VCPU_HRS RSS_GB
printf '%.0s-' {1..60}; echo

awk '
  FNR == 1 { idx=dl=th=ex=rh=vh=rss="" }
  /^task_index/   { idx = $2 }
  /^dl_mpc/       { dl  = $2 }
  /^theta_jn_rad/ { th  = $2 }
  /^exit_code/    { ex  = $2 }
  /^run_hours/    { rh  = $2 }
  /^vcpu_hours/   { vh  = $2 }
  /^peak_rss_gb/  { rss = $2 }
  ENDFILE {
    if (idx != "")
      printf "%-4s %-8.1f %-7.2f %-6s %-10s %-9s %-8s\n", idx, dl, th, ex, rh, vh, rss
    total += vh; n++
    if (rh+0 > max) max = rh+0
  }
  END {
    printf "%.0s-", 1; print ""
    printf "cases %d   mean %.2f vCPU-hr   slowest %.2f h\n", n, (n ? total/n : 0), max
    printf "projected sweep: %.0f vCPU-hr for 100 cases\n", (n ? total/n : 0) * 100
    printf "  on-demand @ $0.041/vCPU-hr: $%.0f\n", (n ? total/n : 0) * 100 * 0.041
    printf "  spot      @ $0.013/vCPU-hr: $%.0f  (x1.5 retry allowance: $%.0f)\n", \
           (n ? total/n : 0) * 100 * 0.013, (n ? total/n : 0) * 100 * 0.013 * 1.5
  }
' "${TMP}"/timing_*.txt

rm -rf "${TMP}"
