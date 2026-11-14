#!/bin/bash
# Submits the one-case Phase 2 benchmark to GCP Batch.
#
#   ./benchmark/submit.sh <bucket-name> [project-id]
#
# Run it from the repository root.
set -euo pipefail

BUCKET="${1:?usage: submit.sh <bucket-name> [project-id]}"
PROJECT="${2:-$(gcloud config get-value project 2>/dev/null)}"
REGION=us-central1
JOB="phase2-bench-$(date +%Y%m%d-%H%M%S)"

if [ -z "${PROJECT}" ] || [ "${PROJECT}" = "(unset)" ]; then
  echo "No project set. Pass one, or run: gcloud config set project <id>" >&2
  exit 1
fi

echo "project ${PROJECT}"
echo "bucket  gs://${BUCKET}"
echo "job     ${JOB}"

# 0. APIs. Harmless if they are already on. This is slow the first time.
gcloud services enable \
  batch.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  logging.googleapis.com \
  --project="${PROJECT}"

# 1. Bucket. Harmless if it already exists.
gcloud storage buckets create "gs://${BUCKET}" \
  --project="${PROJECT}" --location="${REGION}" 2>/dev/null \
  || echo "bucket already exists, reusing it"

# 2. Payload: the sampler script plus the distance marginalization lookup.
tar -czf /tmp/payload.tar.gz \
  bilby_cbc_param_estimation.py \
  .distance_marginalization_lookup.npz

gcloud storage cp /tmp/payload.tar.gz   "gs://${BUCKET}/benchmark/payload.tar.gz"
gcloud storage cp benchmark/run_case.sh "gs://${BUCKET}/benchmark/run_case.sh"

# 3. Let the default compute service account read and write the bucket.
SA="$(gcloud projects describe "${PROJECT}" \
       --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA}" \
  --role=roles/storage.objectAdmin > /dev/null
echo "granted objectAdmin to ${SA}"

# 4. Fill the bucket name into the job spec and submit.
sed "s/__BUCKET__/${BUCKET}/g" benchmark/benchmark_job.json > /tmp/job.json

gcloud batch jobs submit "${JOB}" \
  --project="${PROJECT}" \
  --location="${REGION}" \
  --config=/tmp/job.json

cat <<EOF

Submitted.

Watch the state:
  gcloud batch jobs describe ${JOB} --location=${REGION} --format='value(status.state)'

Confirm all 8 tasks landed on ONE VM:
  gcloud batch jobs describe ${JOB} --location=${REGION} \\
    --format='value(allocationPolicy.instances[0].policy.machineType,status.taskGroups)'

Follow the log:
  UID=\$(gcloud batch jobs describe ${JOB} --location=${REGION} --format='value(uid)')
  gcloud logging read "labels.job_uid=\$UID" \\
    --format='value(textPayload)' --order=asc --limit=200

Read the report when it finishes:
  gcloud storage cat gs://${BUCKET}/benchmark/results/timing_*.txt

Cancel it:
  gcloud batch jobs delete ${JOB} --location=${REGION}
EOF
