Quickstart
==========

This guide will get you started with MORI in 5 minutes.

Basic Communication Example
----------------------------

.. code-block:: python

   import torch
   import mori

   # Initialize MORI
   mori.init_process_group(backend='nccl', world_size=8, rank=0)

   # Create tensor
   tensor = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)

   # All-reduce operation
   mori.all_reduce(tensor, op=mori.ReduceOp.SUM)

   print(f"Reduced tensor shape: {tensor.shape}")

All-Gather Example
-------------------

.. code-block:: python

   import torch
   import mori

   # Input tensor (per rank)
   input_tensor = torch.randn(256, 512, device='cuda', dtype=torch.float16)

   # Output tensor list (all ranks)
   world_size = mori.get_world_size()
   output_tensors = [torch.zeros_like(input_tensor) for _ in range(world_size)]

   # All-gather operation
   mori.all_gather(output_tensors, input_tensor)

   print(f"Gathered {len(output_tensors)} tensors")

Profiling Example
-----------------

.. code-block:: python

   import mori

   # Enable profiler
   mori.profiler.start()

   # Your communication operations
   mori.all_reduce(tensor, op=mori.ReduceOp.SUM)
   mori.barrier()

   # Stop profiler and save results
   mori.profiler.stop()
   mori.profiler.save_results("mori_profile.json")

Performance Tips
----------------

1. **Use FP16/BF16**: Reduce bandwidth requirements
2. **Batch Operations**: Combine multiple small communications
3. **Profile First**: Use MORI profiler to identify bottlenecks
4. **Enable RDMA**: For multi-node setups

Next Steps
----------

* :doc:`MORI-IO-INTRO` - Understand MORI I/O architecture
* :doc:`MORI-IO-BENCHMARK` - See performance benchmarks
* :doc:`PROFILER` - Learn about profiling tools
