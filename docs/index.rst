MORI Documentation
==================

**MORI** (Memory-Optimized ROCm Infrastructure) is AMD's high-performance communication library for distributed AI training and inference on ROCm platforms.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Features

   MORI-IO-INTRO
   MORI-IO-BENCHMARK
   MORI-EP-BENCHMARK
   PROFILER

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/communication
   api/profiler

Features
--------

* **High-Performance I/O**: Optimized communication primitives for ROCm
* **Distributed Training**: Efficient all-reduce, all-gather, and reduce-scatter operations
* **Profiling Tools**: Built-in profiler for performance analysis
* **ROCm Integration**: Native support for AMD Instinct GPUs

Supported GPUs
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - GPU
     - Architecture
     - Memory
     - Status
   * - AMD Instinct MI300X
     - CDNA 3 (gfx942)
     - 192 GB HBM3
     - ✅ Fully Supported
   * - AMD Instinct MI250X
     - CDNA 2 (gfx90a)
     - 128 GB HBM2e
     - ✅ Fully Supported
   * - AMD Instinct MI300A
     - CDNA 3 (gfx950)
     - 128 GB HBM3
     - 🧪 Experimental

Quick Links
-----------

* **GitHub**: https://github.com/ROCm/mori
* **ROCm Documentation**: https://rocm.docs.amd.com
* **Issues**: https://github.com/ROCm/mori/issues

Getting Help
------------

* **Documentation**: https://sunway513.github.io/mori/
* **GitHub Issues**: https://github.com/ROCm/mori/issues
* **ROCm Community**: https://github.com/ROCm/ROCm/discussions

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
