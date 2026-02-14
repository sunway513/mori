Profiler
========

MORI includes a built-in profiler for analyzing communication performance.

Starting the Profiler
----------------------

.. code-block:: python

   import mori

   # Start profiling
   mori.profiler.start()

   # Your code here
   mori.all_reduce(tensor)

   # Stop profiling
   mori.profiler.stop()

Saving Results
--------------

.. code-block:: python

   # Save as JSON
   mori.profiler.save_results("profile.json")

   # Save as CSV
   mori.profiler.save_results("profile.csv", format="csv")

Analyzing Results
-----------------

Profile data includes:

* Operation name
* Duration (microseconds)
* Tensor size (bytes)
* Bandwidth (GB/s)
* Timestamp

**Example output:**

.. code-block:: json

   {
     "operations": [
       {
         "name": "all_reduce",
         "duration_us": 125.3,
         "size_bytes": 4194304,
         "bandwidth_gbps": 31.8,
         "timestamp": 1234567890
       }
     ]
   }

Context Manager
---------------

Use as a context manager for automatic start/stop:

.. code-block:: python

   with mori.profiler.profile():
       mori.all_reduce(tensor)
       mori.barrier()

   # Results automatically saved

Advanced Features
-----------------

**Filter by operation:**

.. code-block:: python

   mori.profiler.filter(operations=["all_reduce", "all_gather"])

**Set output directory:**

.. code-block:: python

   mori.profiler.set_output_dir("/path/to/profiles")

**Enable detailed logging:**

.. code-block:: python

   mori.profiler.set_log_level("DEBUG")

Visualization
-------------

MORI profiler output can be visualized with standard tools:

.. code-block:: bash

   # Convert to Chrome trace format
   python -m mori.profiler.convert profile.json --format chrome

   # View in chrome://tracing
