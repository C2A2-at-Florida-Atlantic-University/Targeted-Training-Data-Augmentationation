# Targeted-Training-Data-Augmentation
Targeted Training Data Augmentation for Under-sampled Subspaces in Machine Learning Models


Many real-world datasets consist of classes with regions of feature space sparsity, where data samples are sparse, rare, or poorly represented. Such regions can limit the ability of a model to learn reliable decision boundaries, hence impairing a model's capacity to generalize. Conventional augmentation approaches generate
synthetic samples uniformly across the dataset, without explicitly
addressing these sparse regions. 
We propose a novel targeted 
augmentation approach that operates on a class-by-class basis, and uses $L_1$-norm principal component analysis to characterize intra-class feature space
geometry. The method identifies non-core (sparse) regions and
allocates synthetic samples preferentially to these areas. The 
method automatically: 1)~computes a class-specific 
augmentation budget, and 2)~allocates synthetic samples, 
generated via different augmentation techniques, 
proportionally to the degree of intra-class data 
scarcity. The proposed approach is data-driven and automated, requiring
no manual parameter tuning, and is compatible with multiple
synthetic data generation techniques. Extensive experiments on
real-world datasets demonstrate that the proposed method
consistently improves classification performance relative to
baseline and generic augmentation strategies, particularly in
data-sparse regions.
