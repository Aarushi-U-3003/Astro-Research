# GCP Compute plan for the Phase 2 sweep

Date: 2026-11-11

## The problem

Phase 2 runs a 10x10 grid: `luminosity_distance` from 50 to 200 Mpc, `theta_jn` from 0 to
1.57 rad. That is 100 cases. One case takes about 4 hours on a 10-core laptop at `npool=3`.
Run in series, the sweep takes about 400 hours.

## Short answer

Rent 2-4 `t2d-standard-60` VMs in `us-central1`. Run one case per vCPU with `npool=1`. The
sweep finishes in 5-15 hours for about **$60-90 on-demand** or **$20-30 on Spot**.

The total work is fixed, so more cores shorten the wall time at almost no extra cost. The
real limits are the GCP CPU quota and preemption risk, not the price.

Do step 0 first. It may cut the bill by a factor of 10.

---

## Step 0: cut the work before renting anything

### Use the relative-binning likelihood

`bilby_cbc_param_estimation.py` builds a `RelativeBinningGravitationalWaveTransient`, then
samples with `full_likelihood`:

```python
result = bilby.run_sampler(likelihood=full_likelihood, ...)   # line 221
```

Relative binning normally gives a 10-100x speedup. `test_file_mismatch.py` and the
`theta_jn_test` diagnostic loop suggest a known disagreement between the two likelihoods.
Settle that question before spending money. A 4-hour case that becomes a 10-minute case
changes the whole plan.

### Two more free savings

| Change | Effect | Risk |
|---|---|---|
| `distance_marginalization=True` | Large speedup and better convergence | None for this use. Bilby reconstructs `luminosity_distance` samples in post-processing, so the EM reweighting still works. The `.distance_marginalization_lookup.npz` file shows a previous attempt. |
| `sample="acceptance-walk"` with `naccept=60`, `nlive=500` | Often 2-3x faster than `sample="rwalk", walks=30, nlive=1000` | Check the evidence and the posterior against one reference case. |

Benchmark one case with each change. Then scale.

---

## VM family

| Family | vCPU meaning | Approx. on-demand $/vCPU-hr | Comment |
|---|---|---|---|
| **`t2d-standard`** (AMD Milan) | Full physical core, SMT off | ~$0.041 | **Recommended.** Built for scale-out throughput. No hyperthread penalty. x86, so `lalsuite` wheels install without trouble. |
| `c4a-standard` (Google Axion, Arm) | Full physical core | ~$0.039 | Best price per core, but Arm. Verify `lalsuite` and `bilby` on `linux-aarch64` first. Do not debug this under time pressure. |
| `n2d-standard` (AMD) | One hyperthread | ~$0.038 | Looks cheaper per vCPU, but two vCPUs are needed to match one T2D core. Worse in practice. |
| `c4` / `c3d` | Thread by default | Higher | Faster per core, but this workload does not need low latency. |

Prices are approximate for `us-central1` and change over time. Confirm in the GCP pricing
calculator.

Use the **standard** memory ratio (4 GB per vCPU). Measure the RSS of one bilby process on
the laptop first. A 4 s, 1024 Hz BNS run is small, so standard is almost certainly enough.

---

## Sizing and cost

These numbers assume one case costs about 15 vCPU-hours on a cloud core, scaled from 4 hours
at `npool=3` on the laptop. **Measure one case on a real VM and rescale.**

| Plan | Total vCPU | `npool` | Concurrent cases | Wall time | Spot | On-demand |
|---|---|---|---|---|---|---|
| A | 100 (2 x `t2d-standard-48`) | 1 | 100 | ~15 h | ~$19 | ~$62 |
| B | 200 (4 x `t2d-standard-48`) | 2 | 100 | ~8 h | ~$21 | ~$69 |
| C | 400 (7 x `t2d-standard-60`) | 4 | 100 | ~5 h | ~$24 | ~$78 |

Plan C is three times faster than plan A and costs $16 more. `npool=1` wastes no cores on
pool overhead, so it gives the lowest total cost. A higher `npool` buys wall time at a small
efficiency loss. Dynesty scales poorly above `npool=8`, so do not go higher.

Add a few dollars for disk and egress. One 100 GB `pd-balanced` boot disk costs about $0.14
per day. Write the result JSON files to a GCS bucket and download at the end.

---

## Must-do settings

### 1. `clean=True` breaks Spot

Line 223 of `bilby_cbc_param_estimation.py` deletes the resume file and restarts from zero.
A preempted job then loses all progress. Change it:

```python
clean=False,
check_point_delta_t=1800,
```

### 2. Stop BLAS oversubscription

This one matters most. If 60 processes run on a 60-core VM and each spawns 60 BLAS threads,
the machine thrashes. Export these before every run:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

### 3. Request quota early

A personal GCP account often starts with 32-128 vCPUs per region. Plan C needs 400. Request
the increase for `CPUS` in `us-central1` a day or two in advance. Spot has a **separate**
quota, so request both if Spot is wanted.

### 4. Fix the blockers in the driver script

| File | Problem |
|---|---|
| `run_grid.py` | Passes `--npool`, which the argparse in `bilby_cbc_param_estimation.py` does not accept. |
| `run_grid.py` | Writes to `logs/`, which does not exist. The first `open()` call fails. |
| `em_sgrb_postprocessing.py`, `em_host_galaxy_postprocessing.py`, `em_joint_postprocessing.py` | Read from `Phase2/outdir_{case_no}_rslice/`, but the sampler writes to `Phase2/outdir_dl_{N}_th_{X}`. The `_rslice` suffix does not match. |

---

## Concrete plan

Build the environment once, snapshot it as a custom image, then launch N identical VMs from
that image. On each VM, run its slice of the case list with GNU `parallel`:

```bash
# cases_vm1.txt: one "dl theta" pair per line
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
parallel --colsep ' ' -j 60 \
  python bilby_cbc_param_estimation.py --dl {1} --theta {2} \
  :::: cases_vm1.txt
gsutil -m rsync -r Phase2/ gs://your-bucket/Phase2/
```

For Spot without manual restarts, use **GCP Batch** instead. It fans 100 tasks over Spot VMs,
retries preempted tasks, and adds no charge above the Compute Engine cost. It is the right
tool, but it costs an afternoon of learning.

## Recommendation

**On-demand, plan B or C.** Spot saves about $45. At this total cost, a run that cannot be
preempted is worth more than $45.

## Caveat

Every cost number here rests on the unverified estimate of 15 vCPU-hours per case. Run
exactly one case on one `t2d-standard-16` VM before committing. That costs under $1 and gives
the true number.
