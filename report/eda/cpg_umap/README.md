# UMAP projection of CpG kinetics

UMAP projection of normalized kinetics features (IPD + pulse width across a 32-base context window) for methylated vs unmethylated CpG sites. Compares to the PCA projection in `cpg_pca_v2/`.

## Why we don't expect class separation

Neither PCA nor UMAP should cleanly separate methylated from unmethylated samples in raw kinetics space. The reason is that **sequence context dominates the feature space**.

Each CpG site has different flanking nucleotides, and the flanking sequence drives most of the kinetics variation — a CpG flanked by `AACGT` has very different baseline IPD/PW than one flanked by `TTCGA`, regardless of methylation status. This sequence-context variance is much larger than the methylation perturbation, so it dominates both global structure (PCA) and local neighborhood structure (UMAP).

Methylation produces a small systematic shift in IPD/PW at and near the modified cytosine, but this shift occurs *within* each sequence context, not as a global displacement. Methylated and unmethylated samples with the same flanking sequence are close neighbors. UMAP builds its graph from local neighborhoods, and those neighborhoods will be mixed-class.

## Why adding nucleotide tokens wouldn't help

Sequence context has no discriminative power for methylation. A CpG site with flanking sequence `AACGT` can be methylated or unmethylated — methylation status is epigenetic, not encoded in DNA. Both classes are drawn from the same CpG sites, so the sequence distribution is identical across classes. Adding sequence tokens would amplify the wrong variance and force UMAP to cluster even more strongly by flanking sequence, pushing the kinetics signal further into the background.

## What the deep model does differently

The supervised model reaches ~80% accuracy because it learns the *interaction* between sequence context and kinetics: "given this specific sequence context, are the kinetics shifted in a way consistent with methylation?" That's a conditional signal. Raw feature projections (PCA, UMAP) operate on marginal distances and can't capture conditional structure.

## Next step: UMAP on learned representations

The right way to get UMAP separation would be to project the **encoder's learned representations** rather than raw features — run samples through the trained encoder and UMAP the hidden states. The encoder has already distilled the conditional kinetics-given-sequence signal into a representation space where classes should separate. This would also serve as a diagnostic for whether the SSL pretraining learns useful representations compared to supervised training.
