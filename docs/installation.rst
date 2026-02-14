Installation
============

Requirements
------------

* Python 3.8 or later
* ROCm 5.7 or later
* PyTorch with ROCm support
* AMD Instinct GPU (MI200 or MI300 series)

Installation Methods
--------------------

From Source
^^^^^^^^^^^

.. code-block:: bash

   # Clone the repository
   git clone --recursive https://github.com/ROCm/mori.git
   cd mori

   # Install dependencies
   pip install -r requirements.txt

   # Build and install
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
   print(f"MORI version: {mori.__version__}")
   print(f"ROCm available: {mori.is_available()}")

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
