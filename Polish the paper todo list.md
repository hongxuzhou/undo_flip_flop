# To-Do List: Polish Workpackages for Submitting to BlackboxNLP 2026. 
- [To-Do List: Polish Workpackages for Submitting to BlackboxNLP 2026.](#to-do-list-polish-workpackages-for-submitting-to-blackboxnlp-2026)
  - [Step 0: Grounding the task](#step-0-grounding-the-task)
  - [Step 1: Must do](#step-1-must-do)
  - [Step 2: Ought to do](#step-2-ought-to-do)
  - [Step 3: Necessary but easy](#step-3-necessary-but-easy)
  - [Step 4: If time and recourse allow](#step-4-if-time-and-recourse-allow)

## Step 0: Grounding the task 
The [BlackboxNLP 2026] (https://blackboxnlp.github.io/2026/) is co-located with EMNLP 2026 in Budapest, Hungary on October 28th, 2026. 
**July 17th, 2026** is the deadline for direct paper submission through openreview. 
Regarding the concepts, this year's workshop specifically states that the topics can include "Examining model performance on simplified or formal languages." -- exactly my project aims to. Furthermore, this year's special tracks is *Reproducibility and Reliability in Interpretability Analyses*. 
The submitted paper should use ACT template, following the formatting requirements. 
This project attempts to submit as an archival paper, meaning that its length is up to 8 pages + references. 

## Step 1: Must do
- [] Multiple seeds and repeated tests (>= 3): Report all the key metrics, including +- std: ID/OOD accuracy, Toggle Heuristic Rate, Deep History Loss Rate, Pressure Test accuracy, etc. 
- [] Transformer baseline: with corresponding result interpretation. Prepare for the three situations (transformers fail too / transformers succeeded, transformers use a different shortcut) -- they cater to different argument paths. 

## Step 2: Ought to do  
- [] Gradient-based attribution test: Compute input * gradient (or, integrated gradients) in read position, compare correct prediction vs. failed cases with respect to the distribution of the attribution weights. For example, are they concentrated around the latest write (toggle) or speared to history write (genuine retrieval). It is doable to Mamba, does not rely on linear methods. 

- [] Small stack ablation (k = 1, 2, 3): Systematically test the model's performance boundaries under different rollback depth. It partially separates the ambiguity between expressibility vs. learnability. For instance, if the model can learn with k = 1 while cannot when k = 2 -> learnability failure; the model consistently fails under different k values -> possibility the upper bound of expressibility. 

## Step 3: Necessary but easy 
- []: Visualisation: fix the transparency of the pie chart: the operation definition of each sector, such as "truly correct", "Right but Heuristic" and more, and the mapping between the original metrics and the pie chart are all missing from the text. They need to be fixed. 
- []: Make the methodological stance of behavioural probing more explicit: Shift the argument of " we stay at the behavioural level due to the nature of SSM" to a more subjective, active choice such as: since the hidden state of SSMs cannot be decomposed like the internal representations of transformers, we deliberately chose archi-free, behaviour-level probing.

## Step 4: If time and recourse allow
- [] Mamba 1 Comparison: Sarrof et al. 2024 is based on Mamba-1. Adding Mamba-1 can better connect the theory and the experiments, and exam if scalar-identity simplification (Mamba-2) is a key variable. 
- [] Curriculum training: gradually increase retraction density, and observe if that represses toggle heuristic. Currently, it is in future work, but it doesn't hurt if we can make it in the current project. 