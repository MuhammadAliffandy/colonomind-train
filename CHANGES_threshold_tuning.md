# What Changed in the Super Agent Routing — and Why

**Affects:** `src/train_dgx.py` only
**Practical impact:** all Intra and Multi results need re-running. Reported numbers will move, most likely downward.

If you only read one section, read §5 — what this means for the paper.

---

## 1. The short version

The Super Agent decides which cases to take over from the CNN by looking at how confident the CNN was. If confidence falls below a cutoff, the case gets routed to the agent. Two things about that setup were wrong.

**The cutoff was a guess.** It was hardcoded at 0.70 and never justified. It is now chosen from held-out data, per model, and the full evidence for the choice is saved to disk.

**The agent was trained on the wrong data.** It learned from the CNN's confidence scores on the training images — the ones the CNN had already memorised. It now learns from a separate held-out slice instead.

Neither of these is the test-set leakage that was fixed earlier in `3b788a0`. These are separate, quieter problems that survived that fix.

---

## 2. Why the old cutoff was a problem

0.70 sounds neutral. It isn't.

A confidence score of 0.70 doesn't mean the same thing in two different models. Our CNNs train with focal loss, which deliberately reshapes how confidence is distributed, and each of the five backbones (ResNet-50, DenseNet-121, EfficientNet-B4, ConvNeXt-Tiny, ViT-B-16) ends up in a different regime. So the same number 0.70 might hand 5% of cases to the agent in one column of our comparison table and 40% in another.

That makes the columns not comparable. When we say "ResNet + agent beats ViT + agent," part of that gap could just be that the two models were operating at different delegation rates — an uncontrolled variable we never reported.

The obvious fix — try several cutoffs and keep the best one — is exactly the thing that got us in trouble the first time. Picking the cutoff by looking at test accuracy is leakage, just smaller and harder to spot than the old feedback loop. So the cutoff has to be chosen using data the model never sees at evaluation time.

---

## 3. Why the agent was learning from the wrong data

This one is subtler and matters more.

The agent's most important input feature is the CNN's confidence. Previously we generated those confidences by running the CNN over its own training images. A network that has seen an image many times is nearly always very sure about it — those confidences cluster up near 1.0.

At test time, on images the CNN has never seen, confidences are spread much lower and much wider.

So the agent was learning "what does a low-confidence case look like?" in a region where it had almost no examples, then being deployed into a region where nearly everything lives. The router was selecting on a feature whose behaviour changes completely between training and use.

Fixing this required a held-out slice the CNN never trained on, which is what §4 sets up.

---

## 4. What the data splits look like now

Before, the training data was cut in two: 80% to train the CNN, 20% held back to decide when to stop training.

That 20% was already busy. Using it *also* to train the agent *and* to pick the cutoff would be asking one holdout to do three jobs — and each job makes it a little less honest as a measure of "unseen data."

So the training pool is now cut three ways:

| Slice | Size | Job |
|---|---|---|
| **train** | 70% | trains the CNN |
| **val_es** | 15% | decides when to stop training — nothing else |
| **val_cal** | 15% | trains the agent and picks the cutoff — nothing else |
| **test** | separate | touched exactly once, at the very end |

The test set is not involved in any decision. It is loaded, predicted on once with everything already frozen, scored, and that's it.

There's one more wrinkle handled inside `val_cal`. If we train the agent on all of `val_cal` and then measure cutoffs on that same `val_cal`, the agent looks better than it is, and we'd pick a cutoff that sends it too much work. So the cutoff is measured using cross-fitting — the agent is repeatedly trained on part of `val_cal` and measured on the part it didn't see. Only after the cutoff is locked does the agent get retrained on the whole slice for actual use.

This costs no extra GPU time. The CNN is still trained once.

---

## 5. What this means for the paper

**Expect the numbers to drop.** Both fixes remove sources of optimism. This is the point — the previous numbers were partly measuring the setup rather than the method.

**Expect the agent to look less impressive.** It may turn out that on some scenarios the agent doesn't help at all. The code handles this explicitly: if no cutoff beats the CNN alone on held-out data, it selects a cutoff of zero, routes nothing, and records `"threshold_source": "degenerate_no_benefit"` in the output. If you see that, it is a real result, not a bug. Report it.

**We can now answer "why 0.70?"** — the question we could not answer before. Every run writes a table of every cutoff tried and how it scored on validation data. That table goes in the supplementary material.

**The framing of the contribution may need to shift.** "The agent beats the CNN" is a claim that may not survive this. "Routing low-confidence cases to a feature-space model recovers accuracy the CNN loses on its uncertain tail" is a claim that probably does, and it's the standard framing in the selective-prediction literature. Worth discussing once we see the re-run numbers.

**All Intra and Multi results must be regenerated.** Old and new numbers cannot be mixed in the same table.

---

## 6. New files you'll see after a run

In each `{model}_Experiment/` folder:

**`{model}_threshold_selection.json`** — the audit trail. The chosen cutoff, how it was chosen, and every cutoff that was tried with its validation score and delegation rate. This is the file to open when someone asks how the cutoff was set.

**`{model}_threshold_sweep_val.png`** — the same information as a chart, measured on validation data, with the chosen point marked.

**`{model}_threshold_sensitivity_test.png`** — the same sweep computed on the test set. **This is shown for transparency only.** It is generated after everything is frozen and never feeds back into any decision. It exists so a reviewer can confirm we didn't pick a lucky spot on the curve. Do not use it to argue for a different cutoff — doing so would recreate the leakage this change removes.

`{model}_metrics.json` keeps all its existing fields and gains a few (chosen cutoff, delegation rate, split sizes, seed, commit hash) so any result can be traced back to the exact run that produced it.

---

## 7. How to run it

No change to the batch scripts — `run_intra_experiments_dgx.sh` and `run_multi_experiments_dgx.sh` work as before. Cutoff tuning is on by default.

To reproduce the old fixed-cutoff behaviour for comparison:

```bash
python train_dgx.py --scenario Intra --train_dataset NTUH --test_dataset NTUH \
  --model ResNet-50 --no_tune_threshold --threshold 0.70
```

Note that this reproduces the old *cutoff*, not the old *numbers* — the split change and the agent fix apply either way.

The batch scripts skip any scenario that already has a `metrics.json`, so **clear or move the old `Result/` folders before re-running**, or nothing will happen.

---

## 8. Known issues this does not fix

**Patient-level grouping.** Our splits are per-image. If several frames come from the same colonoscopy, that patient can appear on both sides of a split, and the model can score well by recognising the patient rather than the disease grade. The data loader doesn't currently track patient or case IDs, so we can't fix this without knowing whether the filenames encode them. **This is the bigger remaining issue** and someone needs to check the NTUH file naming. The LIMUC and TMC-UCM intra-domain runs use those datasets' own official splits and are probably fine.

**`src/train.py` uses the test set for early stopping.** Any number produced by that script is optimistically biased. Separate fix; if you have results from `train.py` in a draft, flag them.

**`scratch_code.py` and `Legacy_Notebooks/`** still contain the old leaky feedback loop. Nothing imports them, but they need deprecation banners so nobody reads them as current practice.
