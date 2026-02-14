Communication Operations
========================

MORI provides high-performance collective communication operations for distributed training.

All-Reduce
----------

.. code-block:: python

   mori.all_reduce(tensor, op=mori.ReduceOp.SUM)

Reduce and broadcast a tensor across all ranks.

**Parameters:**

* **tensor** (*torch.Tensor*) - Input/output tensor
* **op** (*ReduceOp*) - Reduction operation (SUM, PRODUCT, MIN, MAX)

**Example:**

.. code-block:: python

   import torch
   import mori

   tensor = torch.randn(1024, device='cuda')
   mori.all_reduce(tensor, op=mori.ReduceOp.SUM)

All-Gather
----------

.. code-block:: python

   mori.all_gather(output_tensors, input_tensor)

Gather tensors from all ranks.

**Parameters:**

* **output_tensors** (*list[torch.Tensor]*) - List of output tensors
* **input_tensor** (*torch.Tensor*) - Input tensor from this rank

Reduce-Scatter
--------------

.. code-block:: python

   mori.reduce_scatter(output_tensor, input_tensors, op=mori.ReduceOp.SUM)

Reduce and scatter tensors across ranks.

Broadcast
---------

.. code-block:: python

   mori.broadcast(tensor, src=0)

Broadcast a tensor from source rank to all ranks.

**Parameters:**

* **tensor** (*torch.Tensor*) - Tensor to broadcast
* **src** (*int*) - Source rank

Barrier
-------

.. code-block:: python

   mori.barrier()

Synchronize all processes.

Send/Recv
---------

Point-to-point communication:

.. code-block:: python

   # Send
   mori.send(tensor, dst=1)

   # Receive
   mori.recv(tensor, src=0)

Performance Characteristics
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Operation
     - Latency
     - Bandwidth Usage
     - Best For
   * - all_reduce
     - Low
     - High
     - Gradient synchronization
   * - all_gather
     - Medium
     - Very High
     - Data parallelism
   * - reduce_scatter
     - Low
     - Medium
     - Large model training
   * - broadcast
     - Low
     - Medium
     - Parameter distribution
