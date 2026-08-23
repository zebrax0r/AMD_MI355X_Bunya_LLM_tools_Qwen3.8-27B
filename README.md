# Qwen3.8-27B on ONE AMD MI355X — native MXFP4 serving with SGLang

One-click deployment of **Qwen3.8-27B** on a **single** MI355X GPU of UQ's Bunya
supercomputer, with an OpenAI-compatible endpoint, vision input, MTP speculative
decoding, and a benchmark harness.

This is the third repo in a set:

| Repo | Model | GPUs |
|---|---|---|
| [`..._KimiK3`](https://github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_KimiK3) | Kimi K3 | 8 × MI355X |
| [`..._Qwen3.8_MXFP4`](https://github.com/zebrax0r/zebrax0r-AMD_MI355X_Bunya_LLM_tools_Qwen3.8_MXFP4) | Qwen3.8-2.4T-A95B | 8 × MI355X |
| [`..._MI210_..._Qwen3.8-27B`](https://github.com/zebrax0r/AMD_MI210_Bunya_LLM_tools_Qwen3.8-27B) | Qwen3.8-27B (BF16, vLLM) | 2 × MI210 |
| **this one** | **Qwen3.8-27B (MXFP4, SGLang)** | **1 × MI355X** |

---

## Why this repo exists

Getting all eight MI355X GPUs in one allocation is hard to schedule. This repo
answers a different question: **what is the best Qwen3.8 you can serve on a
single card, and how fast can it go?**

---

## The size question, settled

The instinct that there might be something between 27B and 2.4T is a good one.
There isn't. **Qwen shipped exactly two members of the 3.8 family** — the 27B
dense model and the 2.4T-A95B MoE. Everything on the Hub that looks like a
middle rung is a *pruned 2.4T*, and none of them fits 288 GB in a state worth
serving:

| Checkpoint | Size | Verdict |
|---|---:|---|
| `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4` | 1278 GiB | needs ~6 GPUs |
| `INCModel2/…-MXFP4-CT-AutoRound` | 1313 GiB | — |
| `RedHatAI/…-NVFP4-REAP-25` | 1066 GiB | — |
| `nota-ai/…-NVFP4-Global-Pruned-40` | 961 GiB | and NVFP4 needs Blackwell |
| `kernelpool/…-3bit-UVMAX` | 750 GiB | — |
| `mgoin/…-NVFP4-pruned75` | 406 GiB | — |
| `mgoin/…-NVFP4-pruned94` | 170 GiB | **fits** — author: *"basically produces gibberish"* |
| `ckoh04/…-NVFP4-active-slice` | 117 GiB | fits; a slice, not a servable model |

The 2.4T at any usable 4-bit quantisation is ~1.27 TB. A single 288 GB card is
off by a factor of four, and the only variants that close that gap close it by
deleting most of the experts.

So: **27B**, and it is not a consolation prize. It is a 27.8 B-parameter hybrid
with a 262,144-token window, a vision tower, and an MTP head — on a card with
288 GB of HBM3E, which means the whole card is spare capacity for KV and
concurrency.

### Which 27B checkpoint

Decode is **memory-bandwidth-bound**: to emit one token the GPU reads
essentially every weight once. So weight bytes per token is what sets your token
rate, and the ladder is a bandwidth ladder, not a capacity one — all three fit
easily.

| Checkpoint | Size | Format | Ceiling @ 8 TB/s |
|---|---:|---|---:|
| **`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`** | **18.4 GiB** | **MXFP4 W4A4** | **~404 tok/s** |
| `Qwen/Qwen3.8-27B-FP8` | 28.7 GiB | FP8 | ~260 tok/s |
| `Qwen/Qwen3.8-27B` | 51.7 GiB | BF16 | ~144 tok/s |

Those are arithmetic upper bounds, not measurements — `./bench-qwen38-27b.sh
ceiling` prints them and explains how to read them.

**The default is AMD's own Quark AWQ build.** OCP MXFP4: group 32, E8M0 shared
scales, weights *and* activations — precisely the format gfx950 has silicon for.
AMD report GSM8K recovery of **101.8%** (thinking) and **99.2%** (non-thinking)
against the BF16 base, so this is not a quality trade in any measurable sense.

Two properties of that checkpoint matter operationally. Both were verified by
reading its safetensors header directly rather than trusting the model card:

* **The vision tower is excluded from quantisation and stays BF16** — 333
  tensors, 0.86 GiB. It is a full VLM, not a text-only stub. Some community
  quants of the same model drop vision entirely; this one does not.
* **The MTP head ships inside it in BF16** — 15 tensors, including a full
  `self_attn` and `mlp`. So NEXTN speculative decoding costs no extra download,
  no second model to keep in sync, and no meaningful extra HBM.

The quantisation covers the decoder: `linear_attn` projections, MLP gate/up/down,
and the full-attention q/k/v/o. Norms, `A_log`, `dt_bias` and the causal `conv1d`
stay BF16 — which is correct, because the GDN state recursion is where precision
actually matters.

> **Not NVFP4.** Community NVFP4 builds of this model exist and are smaller, but
> NVFP4 is a Blackwell format (E4M3 scales, group 16) and is not what CDNA4
> implements. gfx950 does OCP MXFP4: group 32, E8M0. Use the AMD Quark build,
> which is the same idea in the format this silicon actually has.

---

## Architecture, and where the memory goes

Qwen3.8-27B is a **hybrid**. Of its 64 layers, **48 are Gated DeltaNet** linear
attention and **16 are gated full attention**, in a strict 3:1 pattern
(`full_attention_interval: 4`). The two hold completely different kinds of
memory, and conflating them is how capacity planning goes wrong.

**Full attention — charged per TOKEN.** 24 query heads over 4 KV heads at head
dimension 256:

```
16 layers x 2 (K,V) x 4 heads x 256 x 2 bytes = 65,536 B = 64 KiB/token   (BF16)
                                                           32 KiB/token   (fp8_e4m3)
```

**GDN — charged per REQUEST, and identical whether that request holds 100 tokens
or 100,000.** Per layer: a 10240-wide causal conv of kernel 4, plus a
48 × 128 × 128 SSM state, in fp32 = 3.156 MiB.

```
3.156 MiB x 48 layers = 151.5 MiB per STATE SLOT
```

A request holds **several** slots — 5 under `extra_buffer` (what `auto` resolves
to for this model), 3 under `no_buffer`, 1 with the radix cache off. At 5 slots
that is **0.74 GiB of HBM per concurrent request**.

**The GDN pool, not KV, is what caps concurrency here.** That is the opposite of
the intuition you bring from an all-attention model, and it is why this repo
ships a dedicated command for it.

### `./serve-qwen38-27b.sh budget`

Pure arithmetic — **no GPU, no container, no allocation**. Run it on a login node
while you are deciding what to ask SLURM for:

```
  Card                     288 GB HBM3E (gfx950, CDNA4)
  --mem-fraction-static    0.9
  VLM haircut          NOT applied — MEM_FRACTION is set explicitly

  effective fraction       0.9000
  static pool              259.2 GB
  - weights                 20.0 GB
  = KV + GDN state         239.2 GB

  concurrent        GDN    KV left       total KV      ctx/req
    requests        GiB        GiB         tokens       tokens
  --------------------------------------------------------------------
           8        5.9      216.9      3,552,942     262,144*
          32       23.7      199.1      3,262,062     101,939
          64       47.3      175.4      2,874,222      44,909   <- MAX_RUNNING_REQUESTS
         128       94.7      128.1      2,098,542      16,394
         256      189.4       33.4        547,182       2,137
```

It recomputes for *your* settings, including `KV_CACHE_DTYPE=fp8_e4m3` (doubles
every `ctx/req` figure) and `MAMBA_RADIX_STRATEGY=no_buffer` (5 GDN slots → 3).

### The `--mem-fraction-static` trap, which runs the opposite way to intuition

SGLang applies a ~7.5% haircut to the static fraction for the vision tower
(`adjust_mem_fraction_for_vlm`). But that call sits **inside**
`_handle_gpu_memory_settings()`'s `if self.mem_fraction_static is None:` branch —
so passing an explicit value **skips it entirely**:

| Setting | Effective | Static pool |
|---|---:|---:|
| `MEM_FRACTION=0.9` (default here) | 0.9000 | **259.2 GB** |
| `MEM_FRACTION=auto` | ~0.88 × 0.9248 | 234.4 GB |

**Setting it explicitly gives you more usable HBM, not less.** That is why `0.9`
is the default rather than `auto`, and it is the opposite of what "let the
framework decide" usually implies.

One further reduction *does* apply either way: **`× 0.85` if the attention
backend resolves to `aiter` and context > 8192**. That one lives in
`_handle_attention_backend_compatibility()`, outside the `is None` branch.

Lowering `MEM_FRACTION` to a "safer" `0.85` costs ~14 GB of serving headroom for
nothing. `budget` models every combination, and the server log prints the value
SGLang actually settled on — trust that over any arithmetic here.

> **A quirk worth knowing, on the `auto` path only.**
> `adjust_mem_fraction_for_vlm()` reads `vision_config.num_hidden_layers`. This
> config doesn't define that key — it uses `"depth": 27` — so SGLang falls back
> to its ViT-L/14 default of 24 layers, and the haircut is *milder* than the real
> 27-block tower would justify. Nothing to fix; just don't be puzzled that the
> number isn't 27-based.

### Turning vision off

`ENABLE_MULTIMODAL=0` passes **`--language-model-only`**: the encoder is skipped
entirely, its weights are never loaded, the tower is never built, and multimodal
requests are rejected rather than silently mishandled.

It is deliberately *not* `--enable-multimodal`. SGLang's argument builder
unwraps `Optional[bool]` and renders every bool as `action="store_true"`, so that
flag takes no value and cannot express "off" — `--enable-multimodal False` would
hand argparse a stray positional and fail at startup.


---

## Performance: the two lanes

Two SGLang features apply to the GDN state, both requiring the **Triton**
linear-attention decode backend — which is what ROCm uses anyway, so unlike the
CUDA-only `flashinfer`/`cutedsl`/`nvidia_kda` backends, both are genuinely
available here. They share ring storage with incompatible cursor protocols, so
SGLang **rejects the pair outright**. You pick a lane.

### Lane A — latency (the default)

```
SPECULATIVE=nextn   REPLAYSSM_SPEC=1   REPLAYSSM_DECODE=0
```

MTP speculative decoding, plus `--enable-linear-replayssm-spec`: the spec-verify
ring replaces per-draft full-state snapshots with a per-slot raw-input window, so
the GDN pool admits more concurrent requests at the same ratio. Verify output is
bitwise unchanged — a memory trade, not an accuracy one.

### Lane B — throughput

```
SPECULATIVE=none    REPLAYSSM_SPEC=0   REPLAYSSM_DECODE=1
MAMBA_RADIX_STRATEGY=no_buffer      (set for you if left empty)
```

`--enable-linear-replayssm`: buffered GDN decode, which upstream claims is
**~1.2–1.5× GDN decode bandwidth at batch ≥ 64**. It does nothing for a single
stream — measure it at concurrency or you will conclude it is useless. The
required `no_buffer` strategy is a bonus here: 3 GDN slots per request instead
of 5.

### Which one wins

Nobody knows. There is no published figure for Qwen3.8-27B on gfx950 in MXFP4 on
one card — upstream's verified MI355X cell is for the 2.4T MoE, and their 27B
page is H200/FP8. So measure it:

```bash
./bench-qwen38-27b.sh lanes     # runs A, B, and a no-ReplaySSM control
```

That mode restarts the server three times and records each lane's configuration
in the results file. The control matters: without it you can only tell which lane
beat the *other lane*, not whether either beat plain SGLang.

---

## Speculative decoding is ON by default — and gated

The 2.4T bundle defaults `SPECULATIVE` off, to stay faithful to a verified
upstream cell. **There is no such cell here**, so the default is chosen on merit,
and on merit NEXTN wins: the MTP head is already in the checkpoint, and
single-card decode is bandwidth-bound, which is exactly the regime speculative
decoding exists to fix.

It is not reckless, because ROCm has two speculative-decoding bugs that kill the
whole **server** — not just the offending request:

| Issue | Symptom |
|---|---|
| sglang #32569 | missing HIP `top_k`/`top_p` renorm kernels → `TypeError: 'NoneType' object is not callable` |
| sglang #33694 | `is_hip()` branch claims non-greedy verify but never binds `tree_speculative_sampling_target_only` → `NameError` |

**Qwen's own recommended sampling for this model is `temperature=1.0,
top_p=0.95, top_k=20`** — the documented settings send exactly the affected
parameters. You would not have to go looking for these bugs; a normal client
finds them on the first request.

So `serve-qwen38-27b.sh` probes the image and **refuses to start** rather than
warning. Check any image in about a second, without launching anything:

```bash
./serve-qwen38-27b.sh speccheck
```

Exit codes: `0` safe, `1` unsafe, `2` could not tell (which is treated as
"proceed", because an unreadable probe is not a failed probe). If you have
measured the risk and control every client, `SPEC_GATE=off` starts anyway.

---

## Why SGLang here, when the MI210 sibling uses vLLM

Because gfx950 is a first-class SGLang target and gfx90a is not. Verified
in-tree:

* `Qwen3_5ForConditionalGeneration` is an `EntryClass` in
  `python/sglang/srt/models/qwen3_5.py` — the VLM path, not a text-only variant.
* `quark/schemes/quark_w4a4_mxfp4.py` routes GEMMs through aiter's
  `gemm_afp4wfp4` kernels and **raises** on hardware without FP4 support
  (*"requires gfx95x, e.g. MI355x"*). That gate is the reason the MXFP4 rung is
  faster than FP8 rather than merely smaller.
* The model file carries explicit `_is_hip` / `_use_aiter` paths through
  `Qwen3_5GatedDeltaNet` and the attention layers, including a gfx95-only fused
  all-reduce kernel.
* `model_config.py` rewrites `Qwen3_5ForConditionalGeneration` →
  `Qwen3_5ForCausalLMMTP` for the draft, so NEXTN works on the VLM checkpoint.

**On ROCm, Triton is the only linear-attention backend.** SGLang's `flashinfer`,
`cutedsl`, `nvidia_kda` and `ptx_kda` linear-attention paths are gated behind
`is_cuda()` / `is_sm100_supported()`. Leave `LINEAR_ATTN_BACKEND` empty; there is
nothing to unlock.

---

## Quick start

```bash
# 0. Login node — no allocation needed. Decide your shape.
./serve-qwen38-27b.sh budget
./bench-qwen38-27b.sh ceiling

# 1. Allocate ONE GPU
salloc --partition=admin_test --account=a_rcc --qos=sdf \
       --gres=gpu:mi355x:1 --cpus-per-task=24 --mem=200G --time=2:00:00

# 2. Configure
cp qwen38-27b-env.example qwen38-27b.env
$EDITOR qwen38-27b.env          # set MODEL_CACHE_DIR to /scratch

# 3. Build the image, then check it BEFORE downloading anything
./serve-qwen38-27b.sh pull
./serve-qwen38-27b.sh speccheck   # is speculative decoding safe on this image?
./serve-qwen38-27b.sh gpucheck    # can the container reach the GPU?
./serve-qwen38-27b.sh check       # can this image load this model?

# 4. Weights (~20 GB) and serve
./serve-qwen38-27b.sh download
./serve-qwen38-27b.sh serve --detach

# 5. Prove it actually works
./serve-qwen38-27b.sh toolcheck   # tool calls + reasoning parser
./serve-qwen38-27b.sh vlmcheck    # vision tower, end to end
./bench-qwen38-27b.sh latency
```

Or as a batch job: `mkdir -p logs && sbatch serve-qwen38-27b.sbatch`.

---

## Verifying it works — and why the checks assert on unguessable values

Three failure modes here do **not** produce an error. They produce confident,
fluent, wrong output, and nothing in the stack flags them:

* **A wrong reasoning parser.** Qwen3.8 always reasons; thinking cannot be turned
  off. Get `REASONING_PARSER` wrong and the server still answers `200`, with raw
  `<think>` tags glued to the front of `content`.
* **A vision tower that failed to load.** A VLM that has lost its tower doesn't
  raise — it guesses.
* **A miscompiled GDN kernel.** Linear attention producing garbage state yields
  plausible text, not a crash.

So the checks assert on values the model cannot infer:

* **`toolcheck`** runs a two-turn tool call where the "tool" returns `61.4`, a
  number that appears nowhere in the prompt, and asserts it reaches the final
  answer. It also asserts `reasoning_content` is populated and that no raw
  `<think>` tag leaked into `content`.
* **`vlmcheck`** generates a 128×128 PNG in stdlib — a 4×4 checkerboard with
  exactly one red square, top-right — and asserts the model names that corner.
  The image is *generated*, not fetched, because a check that depends on the
  compute node reaching the internet is a check that fails for the wrong reason.

---

## Bunya specifics

* **The MI355X nodes are `bun159`, `bun160`, `bun161`** — 8 × gfx950, 288 GB HBM
  each.
* **They live in `admin_test`, not `gpu_rocm`.** Recipe:
  `--partition=admin_test --account=a_rcc --qos=sdf --gres=gpu:mi355x:1`.
  Confirm with `sinfo -o "%P %.10l %G %f" | grep mi355x`.
* **`--gres=gpu:mi355x:1` is a request for one GPU, not an exclusive node.**
  Expect to share. Correctness is unaffected (the SLURM cgroup limits what you
  see) but host CPU and the GPFS client are shared, which is worth remembering
  when a benchmark looks noisy. Add `--exclusive` for a clean measurement.
* **Apptainer exists only on compute nodes.** Every mode except `stop`, `status`
  and `budget` must run inside an allocation.
* **Work from `/scratch`**, not `/home` (tight quota) and not RDM. Budget ~50 GB:
  ~20 GB of weights, ~24 GB for the `.sif`.
* **Bunya is GPFS, not Lustre** — no `lfs setstripe`, and `$TMPDIR` is the same
  filesystem.
* `--mem=200G` in the sbatch file is **host RAM**, not HBM.

---

## Repository contents

| File | Purpose |
|---|---|
| `serve-qwen38-27b.sh` | The core script — see the mode list below |
| `serve-qwen38-27b.sbatch` | SLURM batch wrapper (1 GPU) |
| `qwen38-27b-env.example` | Config template — copy to `qwen38-27b.env` and edit |
| `bench-qwen38-27b.sh` | `sweep`, `latency`, `throughput`, `longcontext`, `ceiling`, `lanes` |

Modes of `serve-qwen38-27b.sh`:

| Mode | Needs | What it answers |
|---|---|---|
| `budget` | nothing | how many concurrent requests, at what context |
| `pull` | allocation | build the `.sif` |
| `speccheck` | `.sif` | is speculative decoding safe on this image? |
| `gpucheck` | GPU | can the container reach the GPU? |
| `check` | `.sif` | can this image load this model? checkpoint ladder + sizes |
| `parsers` | `.sif` | which tool-call/reasoning parsers exist |
| `download` | network | prefetch weights |
| `serve` | GPU | run it (`--detach` to background) |
| `toolcheck` | server | tool calls + reasoning parser round trip |
| `vlmcheck` | server | vision tower round trip |
| `loadstat` | log | why the last cold start was slow |
| `stop` / `status` | nothing | lifecycle |

Secrets never live in the repo. `qwen38-27b.env`, the generated API key and the
`.sif` are all gitignored and live on scratch; the key file is written under
`umask 077`; and the server log has `--api-key` and `HF_TOKEN` redacted, because
that log is world-readable on scratch and gets pasted into bug reports.

---

## Things that will bite you

**Sourcing the env file before editing it.** Every line is
`export VAR="${VAR:-default}"`, so an export beats the file — which is what makes
`sbatch --export` and one-off overrides work. It also means that if you `source`
the file, edit it, and `source` again, your edit is silently discarded. This cost
a live debugging session on bun161. Both scripts now **detect** it and print the
`unset` line that fixes it. The cheapest fix is not to source the file at all;
they read it themselves.

**The `MAX_RUNNING_REQUESTS`-48 trap.** With speculative decoding on and this
unset, SGLang's speculative hook pins `--max-running-requests` to **48** — not to
anything derived from memory. Since NEXTN is the default here, that would
silently cap the server. This repo defaults it to **64** for that reason, and
`budget` marks the row.

**`no_buffer` vs the radix cache.** `extra_buffer` is what `auto` resolves to and
is what buys prefix reuse over the mutable GDN state — but it requires the radix
cache. With `DISABLE_RADIX_CACHE=1` the strategy goes inert and the GDN budget
drops from 5 slots per request to 1, a 5× change in concurrency headroom that no
log line mentions.

**Benchmarking without a control.** `lanes` includes one for a reason.

**NEXTN cannot run on the MXFP4 checkpoint, and no flag fixes it.** Seen on
bun160, 23 Aug 2026:

```
parameter.py:223 in load_merged_column_weight
  assert param_data.shape == loaded_weight.shape       # [34816,2560] vs [17408,5120]
```

`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` quantizes its body to MXFP4 but ships its MTP
head in **BF16** — and its `quantization_config` declares only 111
`model.visual.*` entries plus `lm_head` as excluded. `mtp.*` is **not** declared.
So SGLang builds a quantized MTP layer and meets BF16 weights.

`--speculative-draft-model-quantization unquant` looks like the fix and is not.
It sets the draft's quantization to `None`, and `model_config.py` immediately
undoes it:

```python
# Verify quantization configurations.
if self.quantization is None:
    self.quantization = quant_method    # re-detects "quark" from config.json
```

The **target** has an explicit-unset guard (`_quantization_explicitly_unset`);
the **draft** path has none. So the draft is rebuilt quantized every time.

This repo now detects the mismatch before launch and **turns speculative
decoding off**, rather than failing after a weight load. Detection reads the
checkpoint's safetensors header when available and falls back to `config.json`
(always present, even mid-download) — if a quantized checkpoint does not list
`mtp.*` as excluded, NEXTN cannot load it.

**If you want NEXTN**, use a checkpoint whose MTP head matches its body:

| Checkpoint | MTP head | NEXTN |
|---|---|---|
| `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` | BF16, undeclared | ✗ |
| `Qwen/Qwen3.8-27B-FP8` | FP8 with `weight_scale_inv` | ✓ |
| `Qwen/Qwen3.8-27B` | BF16, model is BF16 | ✓ |

`SPEC_ALLOW_MIXED=1` overrides the guard and fails at weight load instead.

**A stray directory named like the repo id shadows the repo id.** The sequel to
the presharded failure below, seen on bun160 minutes later:

```
ValueError: Unrecognized model in amd/Qwen3.8-27B-Quark-AWQ-MXFP4.
Should have a `model_type` key in its config.json.
```

…for a model whose `config.json` is fine, in a setup that had loaded the same
weights successfully minutes earlier. Both transformers and SGLang resolve a
model path by asking `os.path.isdir(model_name_or_path)` **first** and only
falling back to the hub. So a directory literally named
`./amd/Qwen3.8-27B-Quark-AWQ-MXFP4` in the working directory silently beats your
cached snapshot — and since the presharded loader had created it containing only
`presharded/`, there is no `config.json` inside.

One bad `presharded` start therefore poisons every later start until the
directory is removed:

```bash
rm -rf ./amd
```

This repo now refuses to start when such a directory exists without a
`config.json`, and names the `rm` for you. A directory that *does* contain a
`config.json` is a legitimate local checkpoint and is used as-is.

**`LOAD_FORMAT=presharded` breaks the speculative draft.** Seen on bun160,
23 Aug 2026 — the target model loads fine, then:

```
RuntimeError: Cannot find any model weights with `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`
```

…for a checkpoint that is cached and that the target *just loaded*. In
`_first_time_load_and_dump`:

```python
self._ensure_presharded_dir_writable(presharded_dir)   # os.makedirs(...)
...
self.load_weights_and_postprocess(...)                 # -> _prepare_weights()
```

With no `presharded_path` override the root is
`os.path.join(model_path, "presharded")`, and the **draft's** `model_path` is
still the bare repo id — so step 1 creates a *relative* directory
`./amd/Qwen3.8-27B-Quark-AWQ-MXFP4/presharded/…` in your CWD, and step 2's
`is_local = os.path.isdir(model_name_or_path)` is then `True`. The repo id gets
treated as a local model directory, the glob for `*.safetensors` finds only
`presharded/`, and it raises. The target escapes because SGLang rewrites *its*
path to the absolute snapshot ("Found local HF snapshot …"); the draft's is not.

**Leave `LOAD_FORMAT` empty.** Presharding exists for the sibling repo's
1.4 TB / 213-shard / TP8 load. Here the checkpoint is a single 18.4 GiB file
that loads in ~15 s, so it costs a second full copy on disk and buys nothing.
If you set it anyway, this repo now requires `PRESHARDED_PATH` to be set and
absolute, and gives the draft its own root under it.

**aiter's hardcoded `/tmp/aiter_configs`.** Seen on bun160, 23 Aug 2026. Every
SGLang model module fails to import with

```
Ignore import error when loading sglang.srt.models.<x>:
[Errno 13] Permission denied: '/tmp/aiter_configs/bf16_tuned_gemm.csv.lock'
```

and the model registry comes up **empty** — so `check` says it cannot find
`Qwen3_5ForConditionalGeneration`, which reads like "this image cannot serve the
model" when in fact nothing was tested. Left alone it breaks `serve` too, for the
same reason.

The cause is in aiter, not SGLang and not the image. `aiter/jit/core.py` merges
the per-model tuned-GEMM CSVs into a scratch directory:

```python
config_path = Path("/tmp/aiter_configs/")     # hardcoded, no env var
...
lock_path = f"{new_file_path}.lock"
mp_lock(lock_path, write_config)
```

Every *other* aiter config location is an `os.getenv` with a default; this one
is not. And Apptainer bind-mounts the **host's** `/tmp`, so on a shared node the
first user to import aiter creates `/tmp/aiter_configs` owned by themselves and
everyone else fails to write the lock inside it. `mkdir(exist_ok=True)` succeeds,
which is why the error surfaces one line later on the `.lock` rather than on the
directory.

This repo binds a private directory over that exact path (`AITER_CONFIG_DIR`,
default `$MODEL_CACHE_DIR/aiter-configs`) for every probe *and* for `serve`, and
preflights it inside the container before launch. Nothing else about `/tmp`
changes. If you see the error anyway, set `AITER_CONFIG_DIR` to somewhere on
scratch you own.

---

## Running on other hardware

* **MI300X / MI325X (gfx942, CDNA3).** No hardware MX matmul, so the MXFP4
  default will not run — but this model is small enough that you don't need it.
  Set `MODEL_ID=Qwen/Qwen3.8-27B-FP8` (28.7 GiB, and FP8 *is* native on CDNA3) or
  the BF16 build. Both fit one MI300X. The script detects gfx942 and says so.
* **MI210 / MI250 (gfx90a, CDNA2).** Neither FP8 nor MXFP4, and SGLang's ROCm
  build excludes gfx90a entirely. Use the
  [MI210 sibling repo](https://github.com/zebrax0r/AMD_MI210_Bunya_LLM_tools_Qwen3.8-27B),
  which serves the BF16 checkpoint on vLLM at TP=2.

---

## Upstream drift

The image tag is date-stamped (`v0.5.18-rocm720-mi35x-20260822`). Newer tags
appear most days. To find a current one:

```bash
curl -s "https://hub.docker.com/v2/repositories/lmsysorg/sglang-rocm/tags?page_size=100&ordering=last_updated" \
  | python3 -c "import json,sys;[print(t['last_updated'][:10], t['name']) for t in json.load(sys.stdin)['results']]" \
  | grep mi35x | head
```

**Do not substitute the `mi30x` build** — it targets CDNA3 and a different ROCm,
and the MXFP4 kernels will not load. Test a new tag in a **second** `SIF_PATH`
before switching, and run `speccheck` against it first.

---

## Status of the numbers in this README

Everything about **sizes, shapes and formats** was measured: checkpoint sizes
come from the Hugging Face tree API, tensor dtypes and the presence of the MTP
head and vision tower come from reading the safetensors header over HTTP range
requests, and the layer/head/state arithmetic comes from `config.json`.

Everything about **speed** is either arithmetic (the bandwidth ceilings) or
upstream's own claim (the ~1.2–1.5× ReplaySSM figure). **No throughput number
here has been measured on this hardware**, because that requires the allocation
this repo exists to make easy. When you run it, the results are new information —
record the node and the date, and set `BENCH_REPEATS=3` before concluding that
two configurations differ.
