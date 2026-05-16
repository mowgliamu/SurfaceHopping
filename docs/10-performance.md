# 10. Performance — JIT, vmap, GPU

The whole reason `surfacehop_jax` is built on JAX is that the propagator
becomes one XLA program and an ensemble of trajectories runs as one
batched program. This chapter explains how that scales, what dominates
wall time, and how to use the package on a GPU.

## 10.1 The cost model

For an ensemble of $N$ trajectories propagated for $T$ steps with
$n_\mathrm{el}$ electronic states and $n_\mathrm{dim}$ nuclear
coordinates, the wall-clock cost is roughly:

$$
\text{wall time} \;\approx\; \underbrace{T_\text{compile}}_{\text{one-shot}} \;+\; T \cdot N \cdot c(n_\mathrm{el}, n_\mathrm{dim})
$$

where $c(\cdot)$ is the per-step cost of the propagator on one
trajectory. The compile time is paid once per shape signature; thereafter
the XLA program runs at full throughput. The per-step cost scales as
$O(n_\mathrm{el}^3)$ from the eigendecomposition and $O(n_\mathrm{el}^2 n_\mathrm{dim})$
from the gradient tensor; for typical photochemistry ($n_\mathrm{el}=2$–$5$,
$n_\mathrm{dim}=4$–$30$) these are tiny by modern standards.

What this means in practice (CPU laptop, 2024 hardware):

| problem | $N$ traj × $T$ steps | wall (forward) | wall (forward+backward) |
|---|---|---|---|
| Tully-1, full transmission curve | 500 × 5000 | ~5 s | n/a |
| Pyrazine 4-mode, 120 fs | 500 × 5000 | ~30 s | n/a |
| Pyrazine fit, one gradient eval | 30 × 500 | ~3 s | ~7 s |

GPU performance depends on the device, but transparently runs the same
code: ensembles of 5000+ trajectories at 5000 steps each are
comfortable on a single consumer GPU.

## 10.2 Why JIT helps

`jax.lax.scan` traces the step function **once** and the resulting XLA
program runs the loop natively. Without `scan`, a Python `for` loop
calling `step` would have one Python round-trip per step — the
overhead per step would be ~50 μs and the actual propagator work would
be a few μs, i.e., 90%+ overhead.

With `scan`, you pay Python overhead exactly once (during tracing) and
then the loop is XLA-native. For long trajectories this is the
difference between minutes and seconds.

The `simulate` and `run_ensemble` functions both wrap their internal
loops in `scan`, so as long as you call them at the function level
(not piece-by-piece via raw `step`), you get this for free.

## 10.3 Why vmap helps even more

`jax.vmap` is the second multiplier. The pattern for ensembles is:

```python
final, hist = sh.run_ensemble(H, masses, init, dt, n_steps, key)
```

which internally is

```python
def one_traj(init, k):
    return simulate(H, masses, init, dt, n_steps, k, decoherence_fn)
final, hist = jax.vmap(one_traj)(init_states, keys)
```

`vmap` adds a leading axis to every operation in the traced program;
the entire ensemble runs as one batched call. On CPU you get
vectorisation; on GPU you get hundreds-to-thousands-way parallelism
nearly for free.

**Important:** every field of `init_states` must have a leading
`n_traj` axis. The standard Wigner-sample-and-initialise pattern from
[Chapter 7](07-wigner-sampling.md) produces this naturally:

```python
Q0, P0 = sh.sample_phase_space(key_qp, ..., n_samples=n_traj)
V0 = P0 / masses
init = jax.vmap(lambda q, v: sh.initialize(H, q, v, 1, 2))(Q0, V0)
```

If you ever need to assemble the ensemble by hand, use
`jax.tree.map(lambda x: jnp.stack([x_traj_i for ...], axis=0), template)`
to add the batch axis to every field.

## 10.4 Compile time and trace caching

The first call to a `jit`-compiled function pays the compile cost. The
second call with the **same** argument shapes and dtypes reuses the
cached XLA program. The compile cost scales with the depth of any
nested transformations and the size of the unrolled trace; for the
pyrazine ensemble (30 traj, 500 steps, vmap of scan of expm of eigh)
it's ~10 seconds on a laptop CPU.

Things that *trigger a recompile*:

- Changing `n_steps`. `lax.scan` is shape-polymorphic in the loop
  count, but if you change `n_steps` between calls, JAX has to
  re-trace because the step *count* is part of the type signature.
- Changing `n_traj`. Same reason: the leading vmap axis is part of the
  type signature.
- Switching `decoherence_fn` between `None` and `zhu_truhlar` (or any
  other change to the Python-level structure of the function).
- Changing `dt` if it's a Python scalar (JAX promotes Python floats to
  concrete constants in the trace). Wrap as `jnp.array(dt)` to make it
  traced.
- Changing dtype (e.g., toggling `x64`).

Things that *don't* trigger a recompile:

- Changing the values of `init`, `key`, or any other JAX-array
  argument while keeping shapes the same.
- Changing the parameters threaded through the Hamiltonian builder
  (this is the whole point of differentiable workflows).

For parameter-fitting loops, lock in the shapes by establishing them
on the first call and reusing them thereafter.

## 10.5 GPU

Everything in `surfacehop_jax` is `vmap`- and `jit`-compatible, so it
runs on whatever device your `jaxlib` is configured for. The way to
move computation to the GPU is to make sure your input arrays live on
the GPU device when you call `run_ensemble`. JAX does this
automatically when the GPU is the default device:

```python
import jax
print(jax.devices())
# [CudaDevice(id=0)]   ← GPU is default

# Build inputs as usual; they go to the GPU.
final, hist = sh.run_ensemble(H, masses, init, dt=1.0, n_steps=5000,
                              key=jax.random.PRNGKey(0))
# Outputs are on the GPU.  hist.population is a GPU array.
```

For multi-GPU runs you'd need `jax.pmap` (or the newer `jax.sharding`
API) wrapped around `run_ensemble`; the package doesn't ship this
because the single-GPU regime is large enough for most model-Hamiltonian
problems. The single-GPU pattern: build an ensemble of $N \cdot n_\text{shards}$
trajectories and run one call; the GPU saturates with $N \gtrsim 1000$
for $n_\mathrm{el}=2$ problems and $N \gtrsim 100$ for $n_\mathrm{el}\geq 5$.

## 10.6 Memory

For an ensemble of $N$ trajectories propagated for $T$ steps, the
`hist` return contains:

- `hist.population` shape `(N, T, n_el)`, float64
- `hist.active_state` shape `(N, T)`, int64
- `hist.hopped`, `hist.frustrated` shape `(N, T)`, bool
- `hist.total_energy` shape `(N, T)`, float64

For a 1000-trajectory × 5000-step × 2-state run, the dominant arrays
are 1000 × 5000 × 2 × 8 bytes = 80 MB for `population` alone. Not
huge, but on GPU you can hit memory limits with bigger ensembles. Two
mitigations:

1. **Don't return the history.** If you only need the final state, you
   can write a thin wrapper around `simulate` that discards the
   `history` output inside the `scan` body. This drops the memory cost
   to $O(N \cdot n_\mathrm{state\_size})$, independent of $T$. The
   package doesn't provide this convenience function because most
   diagnostic plots need the time history; you can add it locally.

2. **Subsample the history.** Write a scan that returns the
   diagnostic only every $k$ steps (still write the full propagator,
   just yield from `body` conditionally). This is a `lax.scan` with a
   custom carry that includes a step counter.

For parameter-fitting workflows that compute one scalar loss per
forward pass, neither matters: discard `hist`, take just the relevant
field of `final`.

## 10.7 When FSSH gets slow

The package is bottlenecked by:

1. **`jnp.linalg.eigh`** on the diabatic Hamiltonian. $O(n_\mathrm{el}^3)$
   per step. For $n_\mathrm{el} \lesssim 10$ this is fast; for
   $n_\mathrm{el} \sim 100$ (some condensed-phase models) it would
   start to dominate.
2. **`jax.scipy.linalg.expm`** on the electronic generator. Also
   $O(n_\mathrm{el}^3)$ per step, but typically with a smaller
   constant than `eigh`.
3. **`jax.jacrev`** of the Hamiltonian. For an LVC this is fast (the
   Hamiltonian is a single einsum + diag). For an ML potential with
   millions of parameters it's the bottleneck and dominates everything
   else.

For LVC problems the propagator is essentially free; for ML-PES
problems the network evaluation dominates and there's not much the
FSSH layer can do about it. In both cases the JIT-compiled scan +
vmap structure is what you want.

## 10.8 Profiling

To see where time is going, wrap a small representative call in
`jax.profiler`:

```python
import jax
with jax.profiler.trace("./tmp_profile"):
    final, hist = sh.run_ensemble(H, masses, init, dt=1.0, n_steps=500,
                                   key=jax.random.PRNGKey(0))
```

The trace can then be loaded in TensorBoard's profiler UI for a
flame-graph view. Alternatively, `%timeit` in a Jupyter notebook is
often enough — call the function once to compile, then `%timeit` to
measure steady-state cost.

## Next

- [11 — Troubleshooting](11-troubleshooting.md) for diagnosing specific
  failure modes.
