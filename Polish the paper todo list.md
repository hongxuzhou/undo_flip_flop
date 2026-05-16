# To-Do List: Polish Workpackages for Submitting to BlackboxNLP 2026. 
- [To-Do List: Polish Workpackages for Submitting to BlackboxNLP 2026.](#to-do-list-polish-workpackages-for-submitting-to-blackboxnlp-2026)
  - [Step 0: Grounding the task](#step-0-grounding-the-task)
  - [Step 1: Must do](#step-1-must-do)
  - [Step 2: Ought to do](#step-2-ought-to-do)
  - [Step 3: Necessary but easy](#step-3-necessary-but-easy)
  - [Step 4: If time and recourse allow](#step-4-if-time-and-recourse-allow)
  - [Workstream](#workstream)
    - [Phase 0 — Local Preparation (target: by ~May 19)](#phase-0--local-preparation-target-by-may-19)
    - [Phase 1 — HPC Job 1: Multi-seed Mamba-2 (target: submit ~May 19, results ~May 26)](#phase-1--hpc-job-1-multi-seed-mamba-2-target-submit-may-19-results-may-26)
    - [Phase 2 — HPC Job 2: Transformer Baseline (target: submit ~May 26, results ~June 2)](#phase-2--hpc-job-2-transformer-baseline-target-submit-may-26-results-june-2)
    - [Phase 3 — HPC Job 3: Stack Ablation (target: submit ~June 2, results ~June 12)](#phase-3--hpc-job-3-stack-ablation-target-submit-june-2-results-june-12)
    - [Phase 4 — Paper Finalization (target: June 12 – July 10; submit July 17)](#phase-4--paper-finalization-target-june-12--july-10-submit-july-17)
    - [Phase 5 — Optional (only if ahead of schedule after Phase 3)](#phase-5--optional-only-if-ahead-of-schedule-after-phase-3)

## Step 0: Grounding the task 
The [BlackboxNLP 2026] (https://blackboxnlp.github.io/2026/) is co-located with EMNLP 2026 in Budapest, Hungary on October 28th, 2026. 
**July 17th, 2026** is the deadline for direct paper submission through openreview. 
Regarding the concepts, this year's workshop specifically states that the topics can include "Examining model performance on simplified or formal languages." -- exactly my project aims to. 
The submitted paper should use ACT template, following the formatting requirements. 
This project attempts to submit as an archival paper, meaning that its length is up to 8 pages + references. 

## Step 1: Must do
- [ ] Multiple seeds and repeated tests (>= 3): Report all the key metrics, including +- std: ID/OOD accuracy, Toggle Heuristic Rate, Deep History Loss Rate, Pressure Test accuracy, etc. 
- [ ] Transformer baseline: with corresponding result interpretation. Prepare for the three situations (transformers fail too / transformers succeeded, transformers use a different shortcut) -- they cater to different argument paths. 

## Step 2: Ought to do  
- [ ] Gradient-based attribution test: Compute input * gradient (or, integrated gradients) in read position, compare correct prediction vs. failed cases with respect to the distribution of the attribution weights. For example, are they concentrated around the latest write (toggle) or speared to history write (genuine retrieval). It is doable to Mamba, does not rely on linear methods. 

- [ ] Small stack ablation (k = 1, 2, 3): Systematically test the model's performance boundaries under different rollback depth. It partially separates the ambiguity between expressibility vs. learnability. For instance, if the model can learn with k = 1 while cannot when k = 2 -> learnability failure; the model consistently fails under different k values -> possibility the upper bound of expressibility. 

## Step 3: Necessary but easy 
- [ ]: Visualisation: fix the transparency of the pie chart: the operation definition of each sector, such as "truly correct", "Right but Heuristic" and more, and the mapping between the original metrics and the pie chart are all missing from the text. They need to be fixed. 
- [ ]: Make the methodological stance of behavioural probing more explicit: Shift the argument of " we stay at the behavioural level due to the nature of SSM" to a more subjective, active choice such as: since the hidden state of SSMs cannot be decomposed like the internal representations of transformers, we deliberately chose archi-free, behaviour-level probing.

## Step 4: If time and recourse allow
- [ ] Mamba 1 Comparison: Sarrof et al. 2024 is based on Mamba-1. Adding Mamba-1 can better connect the theory and the experiments, and exam if scalar-identity simplification (Mamba-2) is a key variable. 
- [ ] Curriculum training: gradually increase retraction density, and observe if that represses toggle heuristic. Currently, it is in future work, but it doesn't hurt if we can make it in the current project. 

---

## Workstream

*Context: one GPU (A100/H100) at a time with variable queue wait; max 7 days per job. Local tasks fill the gaps between HPC jobs. Deadline: July 17, 2026.*

### Phase 0 — Local Preparation (target: by ~May 19)
These are coding and data tasks that must be done before any HPC job is submitted.

- [ ] Refactor the training loop: accept a `--seed` argument, save model checkpoints after convergence, and write all metrics (ID acc, OOD acc, aggressive acc, heuristic rate, deep history loss rate) to a structured results file (e.g. JSON or CSV) per run.
- [ ] Generate UNDO dataset variants for the stack ablation: create three new dataset splits capping rollback depth at k = 1, 2, 3 (adapt `undo_lff_generate_data.py`).
- [ ] Implement a transformer baseline model (GPT-2 style decoder, matched parameter count to the Mamba-2 2-layer model) in the same training framework.
- [ ] Write the HPC submission script (HTCondor) for Phase 1 that loops over seeds and model configurations.

### Phase 1 — HPC Job 1: Multi-seed Mamba-2 (target: submit ~May 19, results ~May 26)
*What runs on the GPU:*
- [ ] Train all 4 Mamba-2 configurations (1L/2L × Std/UNDO FF) for ≥ 3 seeds each.
- [ ] For each converged checkpoint: run aggressive stress test and causal ablation (perturbation A and B).
- [ ] Aggregate output: mean ± std across seeds for all key metrics.

*Local work while Phase 1 is in queue or running:*
- [ ] Draft the paper skeleton: abstract placeholder, §1 Introduction, §2 Task Definition — these sections do not depend on final numbers.
- [ ] Resolve the pie chart conceptual gap: the 41.1%/58.9% split (aggressive test population) and the 37.43% heuristic rate (ID test causal ablation population) come from different evaluations and must not be mixed in one chart — decide whether to merge them under a unified analysis or present as two separate figures with explicit population descriptions.
- [ ] Rewrite the methodological stance on behavioural probing (Step 3, second bullet) — draft the revised framing as a paper paragraph ready to drop into §3.

### Phase 2 — HPC Job 2: Transformer Baseline (target: submit ~May 26, results ~June 2)
*What runs on the GPU:*
- [ ] Train transformer baseline (1L/2L × Std/UNDO FF) for ≥ 3 seeds each.
- [ ] Apply the identical evaluation suite: ID/OOD accuracy, aggressive stress test, causal ablation.

*Local work while Phase 2 is in queue or running:*
- [ ] Integrate Phase 1 results: update all figures with mean ± std; fix bar chart (add error bars); fix pie chart per the decision made during Phase 1.
- [ ] Write §3 Methods and §4 Mamba-2 Results — these are now fully supported by Phase 1 data.

### Phase 3 — HPC Job 3: Stack Ablation (target: submit ~June 2, results ~June 12)
*What runs on the GPU:*
- [ ] Train UNDO FF models (2-layer Mamba-2) on each of the three k-capped datasets (k = 1, 2, 3) for ≥ 3 seeds each.
- [ ] Evaluate ID/OOD accuracy and run causal ablation for each k to track how heuristic rate changes with rollback depth.

*Note: gradient attribution (Step 2, first bullet) is inference-only — load the best Phase 1 checkpoint and compute input × gradient at read positions. This does not need a dedicated HPC job; run it locally using the saved Phase 1 checkpoint, or append it to the Phase 3 script as a lightweight pre-processing step.*

*Local work while Phase 3 is in queue or running:*
- [ ] Integrate Phase 2 (transformer) results: determine which of the three argument paths applies; write §4 Transformer Comparison subsection.
- [ ] Draft §5 Discussion and §6 Conclusion now that the argument path is locked.

### Phase 4 — Paper Finalization (target: June 12 – July 10; submit July 17)
- [ ] Integrate Phase 3 stack ablation results into §4.
- [ ] Integrate gradient attribution results into §4 or §5 as supporting evidence.
- [ ] Final figure pass: consistent style, captions complete, all metrics defined in text.
- [ ] Polish §1 and abstract once the full argument is in place.
- [ ] Proofread, verify ACL template compliance, complete references.
- [ ] Submit on OpenReview by July 17.

### Phase 5 — Optional (only if ahead of schedule after Phase 3)
- [ ] HPC Job 4: Mamba-1 comparison — train Mamba-1 (1L/2L × Std/UNDO FF), same evaluation suite.
- [ ] HPC Job 5: Curriculum training — gradually increase retraction density across training stages.