# MORI Documentation Accuracy Audit Report

**Date:** 2026-02-14
**Auditor:** Claude Sonnet 4.5
**Scope:** Complete factual accuracy check of all documentation

## Executive Summary

This audit identified **~15 critical factual errors** in the MORI documentation. The most severe issue is that **the entire collective communication API documented does not exist in the codebase**. MORI is not a distributed training communication library as the docs suggest - it provides Expert Parallelism, point-to-point I/O, and symmetric memory management.

## Critical Issues

### 1. Installation (`docs/installation.rst`)

#### Issue 1.1: Python Version Mismatch [FIXED ✓]
- **Documentation claims**: Python 3.8 or later
- **Actual requirement**: Python 3.10+ (setup.py line 172)
- **Status**: FIXED

#### Issue 1.2: Non-existent requirements.txt [FIXED ✓]
- **Documentation**: Instructs `pip install -r requirements.txt`
- **Actual**: No `requirements.txt` exists (only `requirements-build.txt`)
- **Status**: FIXED - removed reference, added PyTorch install note

#### Issue 1.3: Non-functional Verification Code [FIXED ✓]
- **Documentation**: Uses `mori.__version__` and `mori.is_available()`
- **Actual**: Neither function exists in mori/__init__.py
- **Status**: FIXED - replaced with working module checks

### 2. API Documentation - Collective Communication (`docs/api/communication.rst`)

#### Issue 2.1: ENTIRE API DOES NOT EXIST [CRITICAL - NOT FIXED]

The documentation describes a PyTorch-like distributed training API that **completely does not exist**:

| Documented Function | Exists? | Notes |
|---------------------|---------|-------|
| `mori.init_process_group()` | ✗ NO | Does not exist |
| `mori.all_reduce()` | ✗ NO | Does not exist |
| `mori.all_gather()` | ✗ NO | Does not exist |
| `mori.reduce_scatter()` | ✗ NO | Does not exist |
| `mori.broadcast()` | ✗ NO | Does not exist |
| `mori.barrier()` | ✗ NO | Does not exist |
| `mori.send()` / `mori.recv()` | ✗ NO | Does not exist |
| `mori.ReduceOp` enum | ✗ NO | Does not exist |

**What MORI Actually Provides:**

1. **MORI-EP (Expert Parallelism)**:
   - `EpDispatchCombineOp` - MoE token routing operations
   - `EpDispatchCombineConfig` - Configuration for EP operations

2. **MORI-IO (Point-to-Point Communication)**:
   - `IOEngine` - RDMA-based P2P communication
   - `IOEngineConfig` - I/O engine configuration
   - Used for KVCache transfer between GPUs

3. **MORI Shmem (Symmetric Memory)**:
   - OpenSHMEM-style symmetric memory APIs
   - `shmem_torch_process_group_init()` - Initialize SHMEM with PyTorch
   - Memory allocation and synchronization primitives

4. **Kernel Profiler**:
   - Warp-level kernel profiling
   - `export_to_perfetto()` - Export traces to Perfetto format

**Impact**: All code examples in communication.rst will fail. The entire file needs to be rewritten.

**Recommendation**: Completely rewrite `docs/api/communication.rst` to document the actual MORI APIs (EP, IO, Shmem).

### 3. API Documentation - Profiler (`docs/api/profiler.rst`)

#### Issue 3.1: Profiler API Completely Wrong [CRITICAL - NOT FIXED]

The documentation describes a high-level profiler API that does not exist:

| Documented Function | Exists? | Actual API |
|---------------------|---------|------------|
| `mori.profiler.start()` | ✗ NO | - |
| `mori.profiler.stop()` | ✗ NO | - |
| `mori.profiler.save_results()` | ✗ NO | - |
| `with mori.profiler.profile()` | ✗ NO | - |
| `mori.profiler.filter()` | ✗ NO | - |
| `mori.profiler.set_output_dir()` | ✗ NO | - |
| `mori.profiler.set_log_level()` | ✗ NO | - |
| CSV export | ✗ NO | - |
| Chrome trace conversion | ✗ NO | Uses Perfetto format |

**What Actually Exists**:
- `mori.kernel_profiler.export_to_perfetto()` - Export kernel traces to Perfetto JSON
- Warp-level kernel profiling (low-level, not operation-level)

**Impact**: All profiler examples will fail

**Recommendation**: Rewrite `docs/api/profiler.rst` to document actual kernel profiler API

### 4. Quickstart (`docs/quickstart.rst`)

#### Issue 4.1: All Examples Use Non-existent API [NOT FIXED]

All three quickstart examples use functions that don't exist:
- Example 1: `mori.init_process_group()`, `mori.all_reduce()`, `mori.ReduceOp.SUM`
- Example 2: `mori.all_gather()`, `mori.get_world_size()`
- Example 3: `mori.profiler.start()`, `mori.profiler.stop()`, `mori.profiler.save_results()`

**Impact**: None of the quickstart code will run

**Recommendation**: Replace with working examples from MORI-EP-GUIDE.md or actual test code

### 5. Index (`docs/index.rst`)

#### Issue 5.1: Misleading Feature Description [NOT FIXED]

**Documentation claims**:
- "Distributed Training: Efficient all-reduce, all-gather, and reduce-scatter operations"

**Reality**:
- MORI is NOT a collective communication library
- Provides Expert Parallelism (EP), Point-to-Point I/O, and Symmetric Memory (Shmem)
- Collective ops should use PyTorch's `torch.distributed`

**Recommendation**: Update feature list to accurately describe MORI's actual purpose

## What MORI Actually Does

Based on the codebase analysis:

1. **MORI-EP**: Expert Parallelism for Mixture of Experts
   - Dispatch tokens to experts
   - Combine expert outputs
   - Optimized for MoE layers in LLMs

2. **MORI-IO**: High-performance point-to-point communication
   - RDMA-based I/O engine
   - KVCache transfer between GPUs
   - Low-latency P2P operations

3. **MORI Shmem**: Symmetric memory management
   - OpenSHMEM-style APIs
   - GPU memory allocation and synchronization
   - Integrates with PyTorch distributed

4. **Kernel Profiler**: Warp-level profiling tools
   - Collect kernel execution traces
   - Export to Perfetto format for visualization

## Recommendations

### Immediate (P0)
1. ✓ Fix Python version requirement (3.8 → 3.10)
2. ✓ Fix installation verification code
3. ✓ Remove non-existent requirements.txt reference

### High Priority (P1) - Requires Complete Rewrite
1. **Rewrite `docs/api/communication.rst`**:
   - Remove all non-existent collective communication APIs
   - Document actual MORI-EP APIs
   - Document MORI-IO APIs
   - Document MORI Shmem APIs

2. **Rewrite `docs/api/profiler.rst`**:
   - Remove non-existent high-level profiler API
   - Document actual `kernel_profiler` module
   - Document `export_to_perfetto()` function

3. **Rewrite `docs/quickstart.rst`**:
   - Replace all examples with working code
   - Show actual MORI-EP usage
   - Show actual MORI-IO usage
   - Show actual kernel profiler usage

4. **Update `docs/index.rst`**:
   - Fix feature description
   - Clarify MORI's actual purpose
   - Add note about using torch.distributed for collectives

### Medium Priority (P2)
1. Generate API docs from existing MORI-EP-GUIDE.md
2. Add tutorials from examples/ directory
3. Document integration with PyTorch distributed

### Low Priority (P3)
1. Add automated doc tests in CI/CD
2. Generate API reference from code docstrings
3. Add architecture diagrams

## Statistics

- **Total issues found**: ~15
- **Critical (entire API missing)**: 2
- **High (wrong requirements/examples)**: 3
- **Medium**: 0
- **Low**: 0

## Files Reviewed

- ✓ `docs/installation.rst`
- ✓ `docs/quickstart.rst`
- ✓ `docs/api/communication.rst`
- ✓ `docs/api/profiler.rst`
- ✓ `docs/index.rst`

## Reference Documentation

For accurate API documentation, refer to:
- `MORI-EP-GUIDE.md` in repository root
- `examples/` directory for working code
- Source code docstrings in `python/mori/`

---

**Report Generated:** 2026-02-14
**Severity:** CRITICAL - Documentation describes a completely different product
**Next Steps:** Complete rewrite of API documentation required
