#!/usr/bin/env bash
set -euo pipefail
python run_generic_ddpm_gan.py --data-path heart.csv --label-col target --config config.yaml > heart.log 2>&1
python run_generic_ddpm_gan.py --data-path cancer_cell.csv --label-col Class --config config.yaml > cancer.log 2>&1
python run_generic_ddpm_gan.py --data-path glioma.csv --label-col Grade --config config.yaml > glioma.log 2>&1

# or 
# #!/usr/bin/env bash
# set -euo pipefail

# python run_generic_ddpm_gan.py --dataset heart --config config.yaml > heart.log 2>&1
# python run_generic_ddpm_gan.py --dataset cancer_cell --config config.yaml > cancer.log 2>&1
# python run_generic_ddpm_gan.py --dataset glioma --config config.yaml > glioma.log 2>&1
