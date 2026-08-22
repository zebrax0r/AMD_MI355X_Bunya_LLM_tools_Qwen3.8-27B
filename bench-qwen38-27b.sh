#!/usr/bin/env bash
#
# bench-qwen38-27b.sh — measure Qwen3.8 serving performance (tokens/sec, TTFT, TPOT)
# with SGLang's bench_serving client, run inside the same .sif against the live
# endpoint. Use it to compare configs (speculative decoding, radix cache, GDN
# state pool) with real numbers instead of guessing. See the README
# "Performance tuning".
#
# Usage:
#   ./bench-qwen38-27b.sh              sweep concurrency 1/8/32/64 (default)
#   ./bench-qwen38-27b.sh latency      single-stream latency only (concurrency 1)
#   ./bench-qwen38-27b.sh throughput   saturate at BENCH_MAX_CONCURRENCY (64)
#   ./bench-qwen38-27b.sh longcontext  100k in / 512 out, concurrency 1
#   ./bench-qwen38-27b.sh ceiling      the bandwidth bound — arithmetic, no server
#   ./bench-qwen38-27b.sh lanes        A/B the two ReplaySSM lanes + a control
#                                      (RESTARTS the server three times)
#
# Tunables (env or qwen38-27b.env): BENCH_INPUT_LEN, BENCH_OUTPUT_LEN, BENCH_NUM_PROMPTS,
#   BENCH_CONCURRENCY (space-separated list for sweep), BENCH_MAX_CONCURRENCY,
#   BENCH_REPEATS, BENCH_EXTRA_ARGS (appended verbatim to sglang.bench_serving).
#
# 'longcontext' exists because every other shape here is 1024/512, and the
# agentic clients this repo is built for resend 100k+ every turn. Those are
# different regimes, not the same regime scaled. It only changes DEFAULTS — an
# explicit BENCH_* still wins.
#
# THERE IS NO PUBLISHED REFERENCE AT ALL FOR THIS COMBINATION. Upstream has a
# verified MI355X cell for the 2.4T MoE, and an H200/FP8 page for the 27B — but
# nothing for Qwen3.8-27B on gfx950, in MXFP4, on one card. Every number this
# produces is new information rather than a confirmation.
#
# So start with './bench-qwen38-27b.sh ceiling'. It computes the bandwidth bound
# from weight bytes and HBM speed, which is the only fixed reference point you
# have. Then measure against it, record the node and the date with every result,
# and set BENCH_REPEATS=3 before concluding that two configs actually differ.
#
# Runs as a pure HTTP client — no GPU needed — but Apptainer only exists on the
# compute node, so run this from a shell on the serving node (e.g. via
# `srun --overlap --jobid <jobid> --pty /bin/bash -l`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="sweep"
for arg in "$@"; do
    case "$arg" in
        latency|throughput|sweep|longcontext|ceiling|lanes) MODE="$arg" ;;
        *) printf 'Unknown argument: %s (use latency | throughput | sweep | longcontext | ceiling | lanes)\n' "$arg" >&2; exit 1 ;;
    esac
done

# Capture the shape the CALLER asked for, before qwen38-27b.env is sourced.
# qwen38-27b-env.example ships 'export BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"',
# so every real config file sets all of these unconditionally. Once it has been
# sourced, "${BENCH_INPUT_LEN:-100000}" can no longer tell a deliberate 1024
# from a config-file default — which would silently run 'longcontext' at
# 1024/512. A named shape mode has to outrank the config file.
for _v in BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_NUM_PROMPTS BENCH_CONCURRENCY; do
    declare "CLI_$_v=${!_v:-}"
done

log()  { printf '\033[1;34m[bench]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bench WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[bench ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Load config (shared with serve-qwen38-27b.sh) ───────────────────────────────

ENV_FILE="${Q27X_ENV:-$SCRIPT_DIR/qwen38-27b.env}"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

# The pre-source capture above is necessary but NOT sufficient. Anyone who has
# run 'source qwen38-27b.env' — for MODEL_CACHE_DIR in their own shell commands — has
# the whole config file in their environment, so the capture records the sweep
# shape as though the caller had typed it. All four values arriving together is
# the signature. (This is the same hazard the shadow detector below reports for
# the non-BENCH_* variables; here it is handled by value comparison instead,
# because a BENCH_* override is legitimate and common.)
#
# By value alone the two cases are indistinguishable. So ask the config file what
# IT sets, in a subshell with these four unset, and treat an exact match as "this
# came from the config file". A caller passing precisely the configured value
# while also naming a shape mode is contradicting themselves; any other value
# still wins, so 'BENCH_INPUT_LEN=32768 ./bench-qwen38-27b.sh longcontext' is intact.
if [[ -f "$ENV_FILE" ]]; then
    cfg_shape="$(env -u BENCH_INPUT_LEN -u BENCH_OUTPUT_LEN \
                     -u BENCH_NUM_PROMPTS -u BENCH_CONCURRENCY \
        bash -c 'source "$1" >/dev/null 2>&1
                 printf "%s\n%s\n%s\n%s\n" "${BENCH_INPUT_LEN:-}" \
                     "${BENCH_OUTPUT_LEN:-}" "${BENCH_NUM_PROMPTS:-}" \
                     "${BENCH_CONCURRENCY:-}"' _ "$ENV_FILE" 2>/dev/null || true)"
    { read -r CFG_BENCH_INPUT_LEN
      read -r CFG_BENCH_OUTPUT_LEN
      read -r CFG_BENCH_NUM_PROMPTS
      read -r CFG_BENCH_CONCURRENCY
    } <<<"$cfg_shape"
    for _v in BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_NUM_PROMPTS BENCH_CONCURRENCY; do
        _cli="CLI_$_v"; _cfg="CFG_$_v"
        [[ -n "${!_cli:-}" && "${!_cli}" == "${!_cfg:-}" ]] && declare "$_cli="
    done
fi

# The BENCH_* shape is disambiguated above. The rest of the config has the same
# hazard with no defence: sourcing qwen38-27b.env once exports these, and every later
# edit to the file is then silently discarded. Benchmarking one checkpoint while
# the file says another is a particularly bad way to lose an afternoon, so name
# it. Same detector as serve-qwen38-27b.sh; warn, never override.
if [[ -f "$ENV_FILE" ]]; then
    _sv=(SERVED_MODEL_NAME MODEL_CACHE_DIR SIF_PATH MODEL_ID PORT
         SPECULATIVE REPLAYSSM_SPEC REPLAYSSM_DECODE KV_CACHE_DTYPE
         MAX_RUNNING_REQUESTS)
    _ua=(); for _v in "${_sv[@]}"; do _ua+=(-u "$_v"); done
    _fo="$(env "${_ua[@]}" bash -c '
        source "$1" >/dev/null 2>&1
        for v in "${@:2}"; do printf "%s\n" "${!v-}"; done
    ' _ "$ENV_FILE" "${_sv[@]}" 2>/dev/null || true)"
    _shadow=()
    if [[ -n "$_fo" ]]; then
        _i=0
        while IFS= read -r _want; do
            _v="${_sv[$_i]}"; _i=$(( _i + 1 ))
            [[ -n "$_want" && "${!_v-}" != "$_want" ]] && _shadow+=("$_v")
        done <<<"$_fo"
    fi
    if (( ${#_shadow[@]} )); then
        warn "An exported variable is OVERRIDING $ENV_FILE: ${_shadow[*]}"
        warn "  You are benchmarking a configuration the config file does not describe."
        warn "  Fix with:  unset ${_shadow[*]}"
    fi
fi

PORT="${PORT:-30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
SIF_PATH="${SIF_PATH:-}"

# For the 'ceiling' mode these are all irrelevant — it is arithmetic over
# WEIGHTS_GB and needs no container, no .sif and no allocation. Requiring them
# would mean you could not run it on a login node, which is exactly where you
# want it when deciding what to ask SLURM for.
MODEL_ID="${MODEL_ID:-amd/Qwen3.8-27B-Quark-AWQ-MXFP4}"
WEIGHTS_GB="${WEIGHTS_GB:-20}"
[[ "$WEIGHTS_GB" =~ ^[0-9]+$ ]] || die "WEIGHTS_GB must be a positive integer, got '$WEIGHTS_GB'."

if [[ "$MODE" != "ceiling" ]]; then
    # This script already sourced $ENV_FILE above, so "source it" is not the
    # advice — if MODEL_CACHE_DIR is still empty, the file does not set it.
    [[ -n "$MODEL_CACHE_DIR" ]] || die "MODEL_CACHE_DIR is not set. Set it in $ENV_FILE (this script reads that file
     itself — you do not need to source it)."
    SIF_PATH="${SIF_PATH:-$MODEL_CACHE_DIR/qwen38-27b-mi355x.sif}"
    if [[ "$SIF_PATH" == */ || -d "$SIF_PATH" ]]; then
        SIF_PATH="${SIF_PATH%/}/qwen38-27b-mi355x.sif"
    fi
    [[ -f "$SIF_PATH" ]] || die "No .sif at $SIF_PATH (run './serve-qwen38-27b.sh pull' first)."
    command -v apptainer >/dev/null 2>&1 \
        || die "apptainer not found — run this on the serving compute node, not the login node."
fi

# ── ceiling: what is the best this hardware could possibly do? ──────────────
# Pure arithmetic. No server, no container, no GPU, no allocation. Run it
# before you benchmark so you know what "good" would even look like — a
# measured number means very little without the bound it is measured against.
if [[ "$MODE" == "ceiling" ]]; then
    WGB="$WEIGHTS_GB" MID="$MODEL_ID" python3 - <<'CEILPY'
import os
wgb = float(os.environ["WGB"])
mid = os.environ["MID"]
BW = 8.0e12          # MI355X HBM3E, 8 TB/s

print("Single-stream decode ceiling — Qwen3.8-27B on one MI355X")
print("=" * 66)
print("""
  Decode is memory-bound, not compute-bound. To emit ONE token the GPU must
  read essentially every weight once, so the floor on time-per-token is
  (weight bytes / HBM bandwidth). Everything else — kernels, scheduling,
  sampling — only adds to it.

  MI355X HBM3E delivers 8 TB/s. That gives:
""")
print("  %-34s %9s  %9s  %10s" % ("checkpoint", "weights", "ms/token", "tok/s"))
print("  " + "-" * 66)
rungs = [
    ("amd/Qwen3.8-27B-Quark-AWQ-MXFP4", 19.8),
    ("Qwen/Qwen3.8-27B-FP8",            30.8),
    ("Qwen/Qwen3.8-27B",                55.5),
]
if not any(abs(g - wgb) < 0.6 for _, g in rungs):
    rungs.append(("(your WEIGHTS_GB)", wgb))
for name, gb in rungs:
    ms = gb * 1e9 / BW * 1e3
    mark = "  <- configured" if name.split()[0] == mid else ""
    print("  %-34s %6.1f GB  %7.2f ms  %8.0f/s%s" % (name, gb, ms, 1000.0 / ms, mark))
print("  " + "-" * 66)
print("""
  Read these as an UPPER BOUND on single-stream decode, never as a target.
  Real numbers land below them; how far below is the interesting part.

  Two things move you toward the bound rather than away from it:

    speculative decoding (NEXTN, on by default) breaks the one-token-per-
      weight-sweep rule outright — several tokens are verified per sweep, so
      an accepted draft can push measured tok/s ABOVE the figures above. That
      is not an error in the table; it is the entire point of MTP.

    batching does not help single-stream latency at all, but it amortises the
      same weight sweep across many requests, which is why aggregate
      throughput at concurrency 64 is a different measurement entirely.

  So: 'latency' should approach the row above. 'throughput' should exceed it
  by a wide margin. If 'latency' is far below and speculative decoding is on,
  suspect a poor accept rate before you suspect the kernels.
""")
CEILPY
    exit 0
fi


# Benchmark parameters (overridable). Defaults are mode-dependent; anything
# already set in the environment or qwen38-27b.env wins in either branch, so
# 'BENCH_INPUT_LEN=32768 ./bench-qwen38-27b.sh longcontext' does what it looks like.
if [[ "$MODE" == "longcontext" ]]; then
    # n=4, not 200: at 100k a single prefill is measured in tens of seconds, so
    # 200 prompts is an overnight job that nobody will run twice. Four is enough
    # to see TPOT, and TTFT here is a headline number rather than a warmup cost.
    #
    # CLI_* — the pre-source capture — not the post-source values, so that
    # 'BENCH_INPUT_LEN=32768 ./bench-qwen38-27b.sh longcontext' still works while a
    # qwen38-27b.env carrying the sweep shape does not silently override the mode.
    BENCH_INPUT_LEN="${CLI_BENCH_INPUT_LEN:-100000}"
    BENCH_OUTPUT_LEN="${CLI_BENCH_OUTPUT_LEN:-512}"
    BENCH_NUM_PROMPTS="${CLI_BENCH_NUM_PROMPTS:-4}"
    BENCH_CONCURRENCY="${CLI_BENCH_CONCURRENCY:-1}"
else
    BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
    BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-512}"
    BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-200}"
    BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-1 8 32 64}"
fi
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-64}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"

for v in BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_NUM_PROMPTS; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v must be a positive integer, got '${!v}'."
done

# A request longer than the server's context window is refused, and at 100k that
# is an easy way to waste an allocation: CONTEXT_LEN=131072 is this repo's own
# documented fix for KV/GDN allocation failures, and it silently caps what can be
# measured here. Catch it before the first prefill rather than after.
if [[ "${CONTEXT_LEN:-}" =~ ^[0-9]+$ ]] \
   && (( BENCH_INPUT_LEN + BENCH_OUTPUT_LEN > CONTEXT_LEN )); then
    die "This shape needs $((BENCH_INPUT_LEN + BENCH_OUTPUT_LEN)) tokens of context but the
     server was started with CONTEXT_LEN=$CONTEXT_LEN. Either lower BENCH_INPUT_LEN,
     or raise CONTEXT_LEN in qwen38-27b.env and restart the server."
fi

# sglang's random dataset samples each length uniformly from
# [BENCH_*_LEN * ratio, BENCH_*_LEN]. Its own default is 0.0, which means every
# request is on average HALF its nominal size — a nominal 1024/512 sweep then
# actually sends ~507 in / ~262 out per request. Aggregate tok/s at fixed
# concurrency scales with tokens per request, so that quietly halves the headline
# number while leaving TPOT alone, and makes the result incomparable to any
# published figure. Default to 1.0 (fixed sizes, exactly as configured).
BENCH_RANGE_RATIO="${BENCH_RANGE_RATIO:-1.0}"

# How many times to repeat each concurrency. A single run gives you a number
# with no error bar, which is not enough to compare two configurations — or two
# nodes. Set 3-5 when you actually need to tell two things apart; each repeat
# costs a full sweep. Repeats are interleaved per concurrency so slow drift hits
# every repeat alike.
BENCH_REPEATS="${BENCH_REPEATS:-1}"

# ── Resolve API key ─────────────────────────────────────────────────────────

if [[ -z "${Q27X_API_KEY:-}" && -r "$MODEL_CACHE_DIR/q27x-api-key" ]]; then
    Q27X_API_KEY="$(<"$MODEL_CACHE_DIR/q27x-api-key")"
fi
[[ -n "${Q27X_API_KEY:-}" ]] || warn "No API key resolved — bench will 401 (set Q27X_API_KEY)."

# ── Server must be healthy ──────────────────────────────────────────────────

# 'lanes' starts its own servers, so a server being down is its normal state.
if [[ "$MODE" == "lanes" ]]; then
    log "Mode 'lanes' manages the server itself — not requiring one to be up now."
else
    curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
        || die "No healthy server on http://127.0.0.1:${PORT}. Start it with './serve-qwen38-27b.sh serve --detach' first."
    log "Server on port $PORT is healthy."
fi

# ── Reasoning-token warning ─────────────────────────────────────────────────
# Qwen3.8 ALWAYS reasons and reasoning_effort defaults to xhigh, so a nominal
# --random-output-len of 512 is not 512 tokens of answer: most of that budget
# goes to the <think> block, and at xhigh the model may not even finish thinking
# inside it. That does not make the number wrong — it makes it a measurement of
# this model as configured — but it does mean output tok/s here is not
# comparable to a non-reasoning model's, and a short BENCH_OUTPUT_LEN measures
# thinking rather than answering.
log "Note: Qwen3.8 always reasons (reasoning_effort defaults to xhigh), so most"
log "  of BENCH_OUTPUT_LEN=$BENCH_OUTPUT_LEN is <think> tokens. That is a real property of the"
log "  model, not a benchmark artefact — but do not compare these tok/s figures"
log "  against a non-reasoning model's without saying so."

# ── Tuned-MoE detector ──────────────────────────────────────────────────────
# The MXFP4 MoE is the dominant perf lever: on the heuristic FlyDSL fallback it
# runs far below the tuned kernel. Surface which path is active.

LOG_FILE="${LOG_FILE:-$MODEL_CACHE_DIR/q27x-server.log}"
if [[ -r "$LOG_FILE" ]]; then
    if grep -qiE 'no tuned FlyDSL config|heuristic FlyDSL fallback|falling back to.*heuristic' "$LOG_FILE"; then
        warn "MoE is on the SLOW heuristic FlyDSL fallback (no tuned MXFP4 config for these shapes)."
        warn "  Confirm ENABLE_AITER=1 so SGLANG_USE_AITER=1 is set — that is the one env"
        warn "  var the verified MI355X cell requires."
        if [[ "${FLYDSL_FORCE:-0}" == "1" ]]; then
            warn "  FLYDSL_FORCE=1 is forcing gemms through the FlyDSL JIT compiler. If there"
            warn "  is no tuned config for these shapes, that is how you LAND on the"
            warn "  heuristic path. Re-run with FLYDSL_FORCE=0 and compare — that variable"
            warn "  is ours, not upstream's, and this is exactly the case it can cost you."
        fi
    else
        log "No FlyDSL-fallback warning in the server log — tuned MoE path looks active. OK"
    fi
    # Which gemm path was actually taken. FLYDSL_FORCE is a divergence from
    # upstream's cell, so the bench has to say which side of the A/B produced
    # these numbers rather than leaving it to the reader's memory.
    log "aiter env on the run being measured: FLYDSL_FORCE=${FLYDSL_FORCE:-0}, ENABLE_AITER=${ENABLE_AITER:-1}"
    if grep -qiE 'flydsl|jit compil|tuned config' "$LOG_FILE"; then
        log "MoE/gemm path lines from the server log:"
        grep -iE 'flydsl|jit compil|tuned config' "$LOG_FILE" | head -4 | sed 's/^/    /'
    fi
    # A quantisation SGLang resolved differently from what you expected is the
    # other way to be slow without an error. Surface what it actually picked.
    if grep -qiE 'quant(ization)?[ _]?method|Using .*quantization' "$LOG_FILE"; then
        log "Quantisation as resolved by the server:"
        grep -iE 'quant(ization)?[ _]?method|Using .*quantization' "$LOG_FILE" \
            | head -3 | sed 's/^/    /'
    fi
else
    warn "Server log $LOG_FILE not readable; skipping the tuned-MoE check."
fi

# ── Results file ────────────────────────────────────────────────────────────

BENCH_DIR="$MODEL_CACHE_DIR/bench"
mkdir -p "$BENCH_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_FILE="$BENCH_DIR/${STAMP}-${MODE}.txt"
{
    echo "# qwen3.8 benchmark  $(date)"
    # Record the node. On the K3 sibling repo bun161 and bun159 differed by
    # 5-10% on the same image and config, so a result without a hostname cannot
    # be compared against one taken somewhere else.
    echo "# host=$(hostname -s 2>/dev/null || echo '?')  rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo '?')"
    echo "# mode=$MODE model=$SERVED_MODEL_NAME port=$PORT"
    echo "# MODEL_ID=${MODEL_ID:-?}  IMAGE=${SGLANG_IMAGE:-?}"
    echo "# in=$BENCH_INPUT_LEN out=$BENCH_OUTPUT_LEN num_prompts=$BENCH_NUM_PROMPTS range_ratio=$BENCH_RANGE_RATIO"
    echo "# TP_SIZE=${TP_SIZE:-?} DP_SIZE=${DP_SIZE:-?} MEM_FRACTION=${MEM_FRACTION:-?} CONTEXT_LEN=${CONTEXT_LEN:-modelmax}"
    echo "# SPECULATIVE=${SPECULATIVE:-off} MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-auto}"
    echo "# DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-?} MAMBA_RADIX_STRATEGY=${MAMBA_RADIX_STRATEGY:-auto}"
    echo "# MAMBA_FULL_MEMORY_RATIO=${MAMBA_FULL_MEMORY_RATIO:-auto} ENABLE_AITER=${ENABLE_AITER:-?}"
    # The tuning ladder. A result file that does not record which rungs were up
    # cannot be compared against another one, which is the whole point of these.
    echo "# ladder: FLYDSL_FORCE=${FLYDSL_FORCE:-0} ROCM_MULTI_STREAM=${ROCM_MULTI_STREAM:-0}" \
         "AITER_KV_LAYOUT=${AITER_KV_LAYOUT:-nhd} AITER_FP8_PER_TOKEN=${AITER_FP8_PER_TOKEN:-0}" \
         "GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-default}"
    echo "# no published MI355X reference exists for this model — these are first numbers"
    echo
} > "$OUT_FILE"
log "Saving results to $OUT_FILE"

if [[ "$MODE" == "longcontext" ]]; then
    log "Long-context mode: $BENCH_INPUT_LEN in / $BENCH_OUTPUT_LEN out, concurrency $BENCH_CONCURRENCY, n=$BENCH_NUM_PROMPTS."
    log "  Expect a long wait before the first token; that TTFT is itself a result."
    log "  Two thirds of Qwen3.8's layers are GDN linear attention, whose state is"
    log "  O(1) per layer rather than growing with context — so this model should"
    log "  degrade more gently with length than an all-attention model would."
    log "  Whether it does on gfx950 is exactly what this mode measures."
    if [[ -n "${SPECULATIVE:-}" ]]; then
        warn "  SPECULATIVE=$SPECULATIVE. Run this mode again with SPECULATIVE=\"\" and"
        warn "  compare — that pair is the point of the mode. On the K3 sibling the"
        warn "  sign flipped with context (a win at 32k, a loss at 100k), so a short-"
        warn "  context win does not carry. Watch 'accept len' in $LOG_FILE:"
        warn "  below ~2 means the draft is out of its depth, which is a DIFFERENT"
        warn "  failure from the step simply costing too much."
    fi
fi

# ── Which bench module does this image have? ────────────────────────────────
# Upstream moved the implementation to sglang.benchmark.serving and left
# sglang.bench_serving as a shim. PREFER THE OLD PATH WHEREVER IT EXISTS: the
# new module imports disaggregation.utils -> quantization -> aiter at module
# scope, which the old one never touches. This benchmark is a pure HTTP client
# that needs no GPU and no kernels, so the lighter import chain is not just the
# safer choice, it is the correct one. Presence is not usability; only switch
# when there is nothing else to run.
BENCH_MODULE="${BENCH_MODULE:-}"
if [[ -z "$BENCH_MODULE" ]]; then
    BENCH_MODULE="sglang.bench_serving"
    # find_spec, not import: importing drags in torch and prints banners.
    if ! apptainer exec "$SIF_PATH" python3 -c \
         'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("sglang.bench_serving") else 1)' \
         2>/dev/null; then
        BENCH_MODULE="sglang.benchmark.serving"
        warn "This image has no sglang.bench_serving; falling back to $BENCH_MODULE."
        warn "  That module imports aiter at module scope. If it dies in aiter's"
        warn "  jit/core.py, AITER_JIT_DIR is the culprit — see below."
    fi
fi
log "Bench module: $BENCH_MODULE"
echo "# bench_module=$BENCH_MODULE" >> "$OUT_FILE"

# aiter's get_user_jit_dir() branches on `"AITER_JIT_DIR" in os.environ`, NOT on
# whether it has a value, then calls os.makedirs("") on an empty one:
#
#     FileNotFoundError: [Errno 2] No such file or directory: ''
#
# qwen38-27b-env.example ships 'export AITER_JIT_DIR="${AITER_JIT_DIR:-}"', and
# apptainer passes the host environment straight through, so sourcing the config
# is enough to poison any import of aiter inside the container. serve-qwen38-27b.sh
# resolves this to a real path and never sees it. Resolve it the same way here,
# and bind it so the container can actually write there.
AITER_JIT_DIR="${AITER_JIT_DIR:-$MODEL_CACHE_DIR/aiter-jit}"
bench_binds=()
[[ -d "$AITER_JIT_DIR" ]] && bench_binds+=(--bind "$AITER_JIT_DIR")

# ── One benchmark run ───────────────────────────────────────────────────────
# $1 = max concurrency, $2 = request rate ("inf" to saturate)

run_one() {
    local conc="$1" rate="$2"
    log "Run: concurrency=$conc request-rate=$rate${rep:+  repeat $rep/$BENCH_REPEATS}  (in=$BENCH_INPUT_LEN out=$BENCH_OUTPUT_LEN n=$BENCH_NUM_PROMPTS)"
    echo "=== concurrency=$conc request-rate=$rate${rep:+ repeat=$rep} ===" >> "$OUT_FILE"

    # shellcheck disable=SC2086  # intentional splitting of BENCH_EXTRA_ARGS
    apptainer exec \
        --env "OPENAI_API_KEY=${Q27X_API_KEY:-}" \
        --env "AITER_JIT_DIR=$AITER_JIT_DIR" \
        ${bench_binds[@]+"${bench_binds[@]}"} \
        "$SIF_PATH" \
        python3 -m "$BENCH_MODULE" \
            --backend sglang-oai \
            --base-url "http://127.0.0.1:${PORT}" \
            --model "$SERVED_MODEL_NAME" \
            --dataset-name random \
            --random-input-len "$BENCH_INPUT_LEN" \
            --random-output-len "$BENCH_OUTPUT_LEN" \
            --random-range-ratio "$BENCH_RANGE_RATIO" \
            --num-prompts "$BENCH_NUM_PROMPTS" \
            --max-concurrency "$conc" \
            --request-rate "$rate" \
            $BENCH_EXTRA_ARGS \
        2>&1 | tee -a "$OUT_FILE" \
        | grep -iE 'throughput|TTFT|TPOT|ITL|latency|Successful|concurrency|accept' || true
    echo >> "$OUT_FILE"
}

# ── Modes ───────────────────────────────────────────────────────────────────

case "$MODE" in
    latency)    rep=""; run_one 1 inf ;;
    throughput) rep=""; run_one "$BENCH_MAX_CONCURRENCY" inf ;;
    sweep|longcontext)
                for c in $BENCH_CONCURRENCY; do
                    for rep in $(seq 1 "$BENCH_REPEATS"); do run_one "$c" inf; done
                done ;;
    lanes)
        # A/B the two ReplaySSM lanes end to end. This is the one mode that
        # RESTARTS THE SERVER — twice — because the lanes differ in flags that
        # can only be set at launch. It stops whatever is running now.
        #
        # Which lane wins on this hardware is genuinely unknown: upstream
        # publishes nothing for Qwen3.8-27B on gfx950, and the two lanes
        # optimise different things (draft acceptance vs GDN decode bandwidth).
        # The honest answer is to measure both on the shape you actually serve.
        warn "'lanes' will STOP the running server and restart it twice."
        warn "  Each restart is a cold start (~20 GB). Budget the time."
        warn "  Everything else in qwen38-27b.env is held constant across both."
        echo
        LANES_SWEEP="${LANES_SWEEP:-$BENCH_MAX_CONCURRENCY}"
        log "Shape: ${BENCH_INPUT_LEN} in / ${BENCH_OUTPUT_LEN} out, concurrency $LANES_SWEEP, repeats $BENCH_REPEATS"
        echo

        lane_run() {   # $1 = label, rest = VAR=VAL launch overrides
            local label="$1"; shift
            log "════ LANE: $label ════"
            log "  $*"
            "$SCRIPT_DIR/serve-qwen38-27b.sh" stop >/dev/null 2>&1 || true
            if ! env "$@" "$SCRIPT_DIR/serve-qwen38-27b.sh" serve --detach; then
                warn "  Lane '$label' FAILED TO START — skipping it."
                warn "  That is a result too: this lane is not available on this image."
                return 1
            fi
            {
                echo
                echo "########## LANE: $label"
                echo "########## $*"
            } >> "$OUT_FILE"
            local c
            for c in $LANES_SWEEP; do
                for rep in $(seq 1 "$BENCH_REPEATS"); do run_one "$c" inf; done
            done
            "$SCRIPT_DIR/serve-qwen38-27b.sh" stop >/dev/null 2>&1 || true
            echo
        }

        # Lane A — latency: MTP speculative decoding + the spec-verify ring.
        lane_run "A  latency (NEXTN + replayssm-spec)" \
            SPECULATIVE=nextn REPLAYSSM_SPEC=1 REPLAYSSM_DECODE=0 || true

        # Lane B — throughput: no speculation, buffered GDN decode. no_buffer is
        # required by the flag and is set for us, but pass it explicitly so the
        # results file records the configuration rather than implying it.
        lane_run "B  throughput (replayssm-decode, no spec)" \
            SPECULATIVE=none REPLAYSSM_SPEC=0 REPLAYSSM_DECODE=1 \
            MAMBA_RADIX_STRATEGY=no_buffer || true

        # Lane C — the control. Without it you cannot tell whether either
        # ReplaySSM lane beat plain SGLang or merely beat the other lane.
        lane_run "C  control (no spec, no replayssm)" \
            SPECULATIVE=none REPLAYSSM_SPEC=0 REPLAYSSM_DECODE=0 || true

        log "All lanes done. Side-by-side:"
        log "  grep -E 'LANE|Output token throughput|Median TTFT|Median TPOT' $OUT_FILE"
        warn "The server is now STOPPED. Start your chosen lane with:"
        warn "  ./serve-qwen38-27b.sh serve --detach"
        ;;
esac

log "Done. Full output: $OUT_FILE"
if [[ "$BENCH_REPEATS" -le 1 ]]; then
    warn "Single run per point — no error bar. Set BENCH_REPEATS=3 before"
    warn "  concluding that two configs, or two nodes, actually differ."
fi
if [[ -n "${SPECULATIVE:-}" && -z "${MAX_RUNNING_REQUESTS:-}" ]]; then
    warn "SPECULATIVE=$SPECULATIVE with MAX_RUNNING_REQUESTS unset: the speculative hook"
    warn "  pins --max-running-requests to 48, so any concurrency above 48 in this"
    warn "  sweep was queued, not served. Pin it before comparing against a"
    warn "  non-speculative run."
fi
log "Compare runs with:  ls -t $BENCH_DIR"
