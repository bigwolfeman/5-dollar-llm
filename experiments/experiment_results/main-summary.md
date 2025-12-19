Nested optimizer is an MLP being optimized by Adamw.


======================================================================
EXPERIMENT RESULTS SUMMARY
Generated: 2025-12-15T23:00:06.341835
Max Steps: 8000
======================================================================

Experiment                Val Loss     Val Acc      Perplexity   VRAM (GB)    Status    
-------------------------------------------------------------------------------------

## MoE_$5_LLMs

moe_baseline              2.6518       0.4639       14.18        18.23        completed 
moe_nested    nan (3.1 at step 7800, nan by 7900. Read below for more details)
nan was caused by an anti pattern in the GradScaler not checking for inf gradients before stepping into unscale_() 

## Titans

titanmac_baseline                  1.0605       0.7420       2.89         17.29        completed 
titanmac_nested           2.7943       0.4411       16.35        17.93        completed 
======================================================================
DETAILED RESULTS
======================================================================


--------------------------------------------------
Experiment: moe_baseline
Description: MoE + Muon (baseline)
Status: completed
Script: train_moe.py
Arguments:
  max_steps: 8000
  experiment_name: moe_baseline_5000
Runtime: 42.81 minutes
Metrics:
  val_loss: 2.651800
  val_accuracy: 0.463900
  val_perplexity: 14.180000
  peak_vram_gb: 18.230000

--------------------------------------------------
Experiment: moe_nested
Description: MoE + DeepNestedOptimizer
Status: completed
Script: train_moe_nested.py
Arguments:
  max_steps: 8000
  experiment_name: moe_nested_5000
  base_lr: 0.0004603431029542576
  meta_lr: 9.56441656770269e-05
  k_unroll: 1
  momentum_hidden_dim: 128
  controller_hidden_dim: 32
Runtime: 35.77 minutes
Metrics:
  val_loss: nan (3.1 at step 7800, all stats below are from before NaN)
  val_accuracy: 42.94%
  val_perplexity: 18.75
  peak_vram_gb: 20.130000

NaN Log:
The validation metrics were steadily improving right up until the NaN:
  Step 7700: Val Loss: 2.9348, Val Acc: 0.4284
  Step 7750: Val Loss: 2.9336, Val Acc: 0.4293
  Step 7800: Val Loss: 2.9312, Val Acc: 0.4294  ← last good
  Step 7850: NaN

----------------------------------------
Experiment: titanmac
Description: TitanMAC + Muon
Status: completed
Script: train_titanmac.py
Arguments:
  max_steps: 8000
  experiment_name: titanmac_5000
  muon_lr: 0.015533097160640025
  adamw_lr: 0.007022787930614363
Runtime: 57.46 minutes
Metrics:
  val_loss: 1.060500
  val_accuracy: 0.742000
  val_perplexity: 2.890000
  peak_vram_gb: 17.290000


--------------------------------------------------
Experiment: titanmac_nested
Description: TitanMAC + DeepNestedOptimizer
Status: completed
Script: train_titanmac_nested.py
Arguments:
  max_steps: 8000
  experiment_name: titanmac_nested_5000
  base_lr: 0.0004072398672148361
  meta_lr: 9.089757036214503e-05
  k_unroll: 5
  momentum_hidden_dim: 64
  controller_hidden_dim: 16
Runtime: 59.60 minutes
Metrics:
  val_loss: 2.794300
  val_accuracy: 0.441100
  val_perplexity: 16.350000
  peak_vram_gb: 17.930000



depth wasn't touched. Here's the full breakdown:

  Fuzzed:
  | Parameter             | Values Searched    |
  |-----------------------|--------------------|
  | base_lr               | 1e-4 to 3e-3 (log) |
  | meta_lr               | 1e-5 to 5e-4 (log) |
  | k_unroll              | [1, 3, 5]          |
  | momentum_hidden_dim   | [32, 64, 128]      |
  | controller_hidden_dim | [16, 32, 64]       |

  NOT Fuzzed (using defaults):
  | Parameter              | Default          | Description                       |
  |------------------------|------------------|-----------------------------------|
  | MomentumMLP num_layers | 2                | Depth of learned momentum network |
  | NestedController depth | 2 (hardcoded)    | Fixed 2-layer MLP                 |
  | weight_decay           | 0.0              | AdamW weight decay                |
  | max_grad_norm          | 1.0              | Gradient clipping                 |
  | meta_update_freq       | 50               | How often meta-learning runs      |
  | cms_frequencies        | [1] (low_memory) | Multi-timescale memory levels     |
  | min_lr_mult            | 0.1              | LR multiplier floor               |
  | max_lr_mult            | 2.0              | LR multiplier ceiling             |

  Notably: The NestedController has a hardcoded 2-layer architecture (lines 59-64 in nested_controller.py) - no depth parameter exposed at all.

  The MomentumMLP does have num_layers as a parameter (default=2), but it's not wired through to the training scripts.