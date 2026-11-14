#!/bin/bash
# Runs one Phase 2 case on a Batch VM and records how long it takes.
#
# Batch sets BATCH_TASK_INDEX. That index selects a case from CASES below.
# Required environment variables (the Batch bootstrap sets BUCKET and NPOOL):
#   BUCKET  GCS bucket name, with no gs:// prefix
#   NPOOL   dynesty pool workers per case
set -euo pipefail

# Eight points spanning the 10x10 grid. The first four are the corners, so the
# spread between the loudest case (near, face-on) and the quietest one (far,
# edge-on) is measured, not assumed. The rest sample the interior.
CASES=(
  "50.0            0.0"                # 0  nearest, face-on   -> loudest
  "50.0            1.57"               # 1  nearest, edge-on
  "200.0           0.0"                # 2  farthest, face-on
  "200.0           1.57"               # 3  farthest, edge-on  -> quietest
  "116.666666666667 0.697777777777778" # 4  grid midpoint
  "83.333333333333  0.348888888888889" # 5
  "150.0            1.046666666666667" # 6
  "100.0            1.395555555555556" # 7
)

IDX="${BATCH_TASK_INDEX:-4}"
read -r DL THETA <<< "${CASES[${IDX}]}"

# Truncate the distance, do not round it. The sampler names its output with
# Python's int(), which truncates, so %.0f here would disagree with the outdir.
RUN_ID="t${IDX}_dl${DL%%.*}_th$(printf '%.2f' "${THETA}")_npool${NPOOL}"
WORK=/opt/bench
REPORT=/tmp/timing_${RUN_ID}.txt

stamp() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] task ${IDX}: $*"; }

# ---------------------------------------------------------------- environment
# One BLAS thread per process. Without this, all eight tasks each spawn a full
# thread pool on the same eight cores and the VM thrashes. The benchmark must
# match the sweep conditions, or the measurement does not transfer.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg

SETUP_START=$(date +%s)
stamp "setup starts for dl=${DL} theta=${THETA}"

VENV=/opt/bench/venv

# Each task gets its own directory. Eight tasks share this VM, so a shared
# working directory would let them overwrite each other's output.
WORK="${WORK}/task${IDX}"
mkdir -p "${WORK}"
cd "${WORK}"

# All eight tasks run on one VM, so apt and the venv must be built exactly
# once. flock serialises them. Concurrent apt-get calls deadlock on the dpkg
# lock, and eight parallel pip installs would waste 12 GB and ten minutes.
export DEBIAN_FRONTEND=noninteractive
exec 9>/tmp/bench_setup.lock
flock 9

if ! command -v /usr/bin/time > /dev/null 2>&1; then
  stamp "installing system packages"
  apt-get update -qq
  apt-get install -y -qq python3 python3-venv python3-pip time > /dev/null
fi

if [ ! -f "${VENV}/.ready" ]; then
  stamp "building the shared venv with $(python3 --version)"
  python3 -m venv "${VENV}"
  "${VENV}/bin/pip" install --quiet --upgrade pip
  # Pinned to the versions measured on the laptop. The timing only transfers
  # to the sweep if the sampler stack is identical.
  "${VENV}/bin/pip" install --quiet \
    numpy==2.4.6 \
    scipy==1.17.1 \
    pandas==3.0.3 \
    matplotlib==3.11.0 \
    astropy==8.0.1 \
    h5py==3.16.0 \
    corner==2.3.0 \
    dynesty==3.0.0 \
    bilby==2.8.1 \
    bilby.cython==0.5.4 \
    lalsuite==7.26.15
  touch "${VENV}/.ready"
fi

flock -u 9

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
stamp "packages ready"

# The payload holds bilby_cbc_param_estimation.py and the distance
# marginalization lookup table. Shipping the lookup avoids rebuilding it, which
# would otherwise land inside the measured window.
gcloud storage cp "gs://${BUCKET}/benchmark/payload.tar.gz" "/tmp/payload_${IDX}.tar.gz"
tar -xzf "/tmp/payload_${IDX}.tar.gz" -C "${WORK}"

SETUP_END=$(date +%s)
SETUP_SECONDS=$((SETUP_END - SETUP_START))
stamp "setup done in ${SETUP_SECONDS} s"

# ------------------------------------------------------------------ the case
{
  echo "task_index       ${IDX}"
  echo "run_id           ${RUN_ID}"
  echo "dl_mpc           ${DL}"
  echo "theta_jn_rad     ${THETA}"
  echo "npool            ${NPOOL}"
  echo "machine_type     $(curl -s -H 'Metadata-Flavor: Google' \
                             http://metadata.google.internal/computeMetadata/v1/instance/machine-type \
                           | awk -F/ '{print $NF}')"
  echo "cpu_model        $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
  echo "vcpus_on_vm      $(nproc)"
  echo "mem_total_kb     $(awk '/MemTotal/ {print $2}' /proc/meminfo)"
  echo "setup_seconds    ${SETUP_SECONDS}"
} > "${REPORT}"

stamp "sampler starts"
RUN_START=$(date +%s)

# /usr/bin/time -v gives wall clock and peak RSS. Peak RSS decides how many
# tasks fit on one VM in the sweep, so it is the second number we need.
set +e
/usr/bin/time -v -o "/tmp/time_${RUN_ID}.txt" \
  python bilby_cbc_param_estimation.py \
    --dl "${DL}" --theta "${THETA}" --npool "${NPOOL}" \
    > "/tmp/stdout_${RUN_ID}.log" 2>&1
EXIT_CODE=$?
set -e

RUN_END=$(date +%s)
RUN_SECONDS=$((RUN_END - RUN_START))
stamp "sampler finished in ${RUN_SECONDS} s with exit code ${EXIT_CODE}"

PEAK_RSS_KB=$(awk -F': ' '/Maximum resident set size/ {print $2}' \
                "/tmp/time_${RUN_ID}.txt" 2>/dev/null || echo 0)

{
  echo "exit_code        ${EXIT_CODE}"
  echo "run_seconds      ${RUN_SECONDS}"
  echo "run_hours        $(awk -v s="${RUN_SECONDS}" 'BEGIN {printf "%.2f", s/3600}')"
  echo "vcpu_hours       $(awk -v s="${RUN_SECONDS}" -v n="${NPOOL}" \
                             'BEGIN {printf "%.2f", (s/3600)*n}')"
  echo "peak_rss_kb      ${PEAK_RSS_KB}"
  echo "peak_rss_gb      $(awk -v k="${PEAK_RSS_KB}" 'BEGIN {printf "%.2f", k/1048576}')"
} >> "${REPORT}"

# ------------------------------------------------------------------- results
# Upload after the clock stops, so transfer time never enters the measurement.
stamp "uploading"
gcloud storage cp "${REPORT}"                 "gs://${BUCKET}/benchmark/results/" || true
gcloud storage cp "/tmp/time_${RUN_ID}.txt"   "gs://${BUCKET}/benchmark/results/" || true
gcloud storage cp "/tmp/stdout_${RUN_ID}.log" "gs://${BUCKET}/benchmark/results/" || true
if [ -d "${WORK}/Phase2" ]; then
  gcloud storage cp -r "${WORK}/Phase2" "gs://${BUCKET}/benchmark/results/" || true
fi

echo "============ BENCHMARK REPORT task ${IDX} ============"
cat "${REPORT}"
echo "======================================================"

exit "${EXIT_CODE}"
