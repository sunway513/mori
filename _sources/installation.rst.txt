Installation
============

Requirements
------------

* Python 3.10 or later
* ROCm 6.0 or later (ROCm 6.4+ recommended)
* PyTorch with ROCm support
* AMD Instinct GPU (MI200 or MI300 series recommended)

Installation Methods
--------------------

From Source
^^^^^^^^^^^

.. code-block:: bash

   # Clone the repository
   git clone --recursive https://github.com/ROCm/mori.git
   cd mori

   # Install PyTorch with ROCm support first
   # See https://pytorch.org for ROCm installation instructions

   # Build and install MORI
   python3 setup.py develop

Environment Variables
---------------------

Required environment variables:

.. code-block:: bash

   # ROCm installation path
   export ROCM_PATH=/opt/rocm

   # GPU architectures to compile for
   export GPU_ARCHS="gfx90a;gfx942"

   # Enable MORI features
   export MORI_ENABLE_PROFILER=1

Verification
------------

Verify the installation:

.. code-block:: python

   import mori
   import torch

   # Check if modules loaded successfully
   print("MORI modules available:")
   print(f"  - mori.shmem: {hasattr(mori, 'shmem')}")
   print(f"  - mori.ops: {hasattr(mori, 'ops')}")
   print(f"  - mori.io: {hasattr(mori, 'io')}")
   print(f"  - mori.kernel_profiler: {hasattr(mori, 'kernel_profiler')}")

   # Check ROCm availability via PyTorch
   print(f"\nPyTorch version: {torch.__version__}")
   print(f"ROCm available: {torch.cuda.is_available()}")
   print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")

Troubleshooting
---------------

**ImportError: No module named 'mori'**
   Ensure ROCm libraries are in your library path:

   .. code-block:: bash

      export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

**RuntimeError: No AMD GPU found**
   Verify GPU is accessible:

   .. code-block:: bash

      rocm-smi
      rocminfo | grep gfx
