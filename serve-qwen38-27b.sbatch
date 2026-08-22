#!/bin/bash --login
#SBATCH --job-name=q27x-mi355x
#SBATCH --partition=admin_test
#SBATCH --account=a_rcc
#SBATCH --qos=sdf
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi355x:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/q27x-%j.out
#SBATCH --error=logs/q27x-%j.out
#
# Batch wrapper around serve-qwen38-27b.sh for ONE MI355X GPU on Bunya
# (bun159/160/161). Submit from /scratch (not /home, not RDM) with:
#
#   mkdir -p logs && sbatch serve-qwen38-27b.sbatch
#
# ── ONE GPU, AND WHY THAT IS THE WHOLE POINT ───────────────────────────────
#
# The sibling 2.4T bundle needs --gres=gpu:mi355x:8 and cannot run on less:
# 1.4 TB of MXFP4 weights against 8 x 288 GB is the entire node. Holding all
# eight at once is hard to schedule, which is what this repo exists to avoid.
#
# Qwen3.8-27B in MXFP4 is 18.4 GiB — 6.4% of ONE card. So this asks for one,
# queues like a normal job, and leaves the other seven for everyone else.
#
# --gres=gpu:mi355x:1 is a request for one GPU, NOT for an exclusive node, so
# expect to share the node with other jobs. That is fine for correctness (the
# SLURM cgroup limits what you can see) but it does mean host CPU and the GPFS
# client are shared — worth remembering when a benchmark looks noisy. If you
# want a clean measurement, add '#SBATCH --exclusive' and accept the queue.
#
# ── PARTITION / QOS ────────────────────────────────────────────────────────
#
# The partition/account/qos above are the DISCOVERED MI355X recipe, carried
# over from the 2.4T bundle where they were found by running them: the MI355X
# nodes live in admin_test, not gpu_rocm. Confirm the current shape with:
#
#   sinfo -o "%P %.10l %G %f" | grep mi355x
#   sacctmgr show assoc user=$USER format=account,qos,partition
#
# The nodes are bun159/bun160/bun161; the gres lands you on a free one. Add
# '#SBATCH --nodelist=bun161' to pin one.
#
# ── CPU AND MEMORY ─────────────────────────────────────────────────────────
#
# --cpus-per-task=24 is one eighth of the node's 192, matching the one-eighth
# GPU request. It is ample: this checkpoint is a single safetensors file, so
# parallel weight loading has little to parallelise (WEIGHT_LOAD_THREADS
# matters on the FP8 and BF16 rungs, which are sharded).
#
# --mem=200G is HOST RAM for staging the weight load, not HBM. The weights are
# ~20 GB, so this is generous on purpose — it leaves plenty of page cache,
# which is what makes a restart on the same node much faster than a first start.
#
# ── WALLTIME ───────────────────────────────────────────────────────────────
#
# A cold start reads ~20 GB off scratch and captures CUDA graphs. That is
# minutes, not the hours the 2.4T needs, which is why READY_TIMEOUT defaults to
# 1800s here rather than 14400s. It must stay below this walltime.
#
# ── AFTER SUBMIT ───────────────────────────────────────────────────────────
#
# The connection banner (node, endpoint URL, API key location, tunnel command)
# is printed into the job log. Once the job is RUNNING:
#
#   grep -A 30 'is up and serving' logs/q27x-<jobid>.out
#
# The server runs until walltime or 'scancel <jobid>' (the script traps SIGTERM
# and stops the container cleanly).
#
# ── BEFORE YOUR FIRST SUBMIT, in an interactive allocation ─────────────────
#
#   salloc --partition=admin_test --account=a_rcc --qos=sdf \
#          --gres=gpu:mi355x:1 --cpus-per-task=24 --mem=200G --time=2:00:00
#   ./serve-qwen38-27b.sh pull        # build the .sif
#   ./serve-qwen38-27b.sh speccheck   # is speculative decoding SAFE here?
#   ./serve-qwen38-27b.sh check       # can this image load this model?
#   ./serve-qwen38-27b.sh download    # ~20 GB
#
# Run 'speccheck' before anything else. Speculative decoding is ON by default in
# this bundle, and serve-qwen38-27b.sh REFUSES TO START on an image whose HIP
# sampling kernels would make it unsafe. Better to learn that in one second at
# the top of your allocation than at the end of a cold start.
#
# './serve-qwen38-27b.sh budget' needs no allocation at all — run it on a login
# node beforehand to decide MAX_RUNNING_REQUESTS and whether you want fp8 KV.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[q27x-sbatch] Job $SLURM_JOB_ID on $(hostname) at $(date)"
exec ./serve-qwen38-27b.sh serve
