# Targeted Training Data Augmentation for Under-sampled Subspaces

A data-driven, automated augmentation framework that identifies and preferentially fills intra-class sparse regions in feature space, improving model performance in underrepresented regions without manual tuning.

Many real-world datasets contain regions within classes where samples are sparse or poorly represented. These regions limit a model’s ability to learn stable decision boundaries. Traditional augmentation methods generate synthetic data uniformly across the dataset, ignoring these weak regions.

This repository implements a targeted augmentation pipeline that focuses specifically on such under-sampled regions. The method operates on a class-by-class basis, uses L1-norm PCA to capture intra-class geometry, identifies non-core (sparse) samples, and generates synthetic data using TabDDPM. The final augmented dataset is constructed using a geometry-based selection criterion, and evaluation is performed on outlier-only test data to measure improvements in sparse regions.

Key properties of the method include:
- Targeted augmentation focused on sparse regions
- Fully automated pipeline with no manual tuning required
- Modular design supporting diffusion-based generation
- Data-driven allocation based on intrinsic feature-space structure

---

## Requirements

Python 3.10.x is required.

Check your version:
python --version

Install dependencies:
pip install -r requirements.txt

---

## Running the Code

Run the pipeline using dataset names defined in config.yaml:

python run_generic_ddpm_gan.py --dataset heart --config config.yaml  
python run_generic_ddpm_gan.py --dataset cancer_cell --config config.yaml  
python run_generic_ddpm_gan.py --dataset glioma --config config.yaml  

To save output:

python run_generic_ddpm_gan.py --dataset heart --config config.yaml > heart.log 2>&1

To run all datasets:

bash scripts/run_all.sh

---

## Configuration

All parameters are defined in config.yaml. Each dataset has its own configuration block that controls data paths, model selection, scaling, and DDPM settings.

Example:

datasets:
  heart:
    data:
      data_path: heart.csv
      label_col: target
    experiment:
      models: [RF, DT]
    scaling:
      classifier: none
      ddpm: minmax
    ddpm:
      train_steps: 800
    models:
      RF:
        n_estimators: 250
      DT:
        max_depth: 4

---

## Hyperparameter Tuning

Best performance is obtained by tuning parameters per dataset. Modify config.yaml and rerun the pipeline.

Recommended ranges:

LR: C = 0.1, 1.0, 10.0  
SVM: C = 0.1 to 100, gamma = scale or auto  
RF: n_estimators = 100 to 500  
DT: max_depth = 3, 5, 8 or null  
DDPM: increase train_steps (e.g., 800 to 2000), use minmax scaling  

Each execution runs a single configuration. For grid search, use simple scripts or loops.

---

## Notes

The pipeline performs targeted augmentation only and evaluates performance on outlier-only test data. Dataset CSV files must be present in the project directory or paths must be updated in config.yaml. The supporting modules are compiled, so Python 3.10.x is required.

---

## Results

The method consistently improves classification performance compared to no augmentation and uniform augmentation approaches. Gains are most pronounced in sparse regions of the feature space.

---

## Citation

Coming soon.
