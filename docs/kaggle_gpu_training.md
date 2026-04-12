# Kaggle GPU Training Guide

## Goal

Train the anomaly pipeline on Kaggle GPU and export artifacts safely for GitHub workflow integration.

## Recommended Dataset

Primary dataset for this project:

- Kaggle: UCF Crime Dataset (odins0n/ucf-crime-dataset)
- Why: aligns with real surveillance incidents and public-safety classes used in this project.
- Reported content on Kaggle page: 14 classes, approximately 1.38M frame images, about 11.57 GB.

Alternative when you want original long videos instead of extracted frames:

- Kaggle: UCF_Crimes (bypktt/ucf-crimes), around 103 GB.
- Trade-off: closer to original benchmark video setup, but much heavier for Kaggle session limits.

## Recommended Kaggle Setup

1. Open a new Kaggle Notebook.
2. Enable GPU accelerator.
3. Turn Internet on.
4. Add dataset input: odins0n/ucf-crime-dataset.

## Notebook Steps

You can run either:

- Notebook workflow: notebooks/kaggle_gpu_training.ipynb
- Command workflow: the steps below

1. Clone your repository.

   Example command:
   git clone YOUR_GITHUB_REPO_URL
   cd YOUR_REPO

2. Install project dependencies for training.

   Example command:
   pip install -r ml/requirements.txt

3. Run Kaggle training orchestrator.

   Example command:
   python scripts/kaggle_train.py --input-dir /kaggle/input/ucf-crime-dataset --output-dir /kaggle/working/incident-intel-output --device auto --epochs 40 --latent-dim 64 --batch-size 128 --max-images 300000

   Note: webcam benchmark is disabled by default in Kaggle. If you want benchmark results, add --run-benchmark.

4. Verify output files under /kaggle/working/incident-intel-output:

   - features.csv
   - autoencoder.pt
   - isolation_forest.joblib
   - eval_report.json
   - training_summary.json
   - incident_intel_training_outputs.zip

5. Download incident_intel_training_outputs.zip from Kaggle output panel.

## Push Strategy For GitHub

Your repository already ignores heavy generated artifacts using .gitignore, which is the correct default.

Recommended push flow:

1. Push only source code and configs.
2. Do not commit model binaries or generated datasets directly.
3. Store trained artifacts in one of these:

   - GitHub Release assets
   - DVC remote storage
   - Cloud storage bucket

4. Commit evaluation summaries for reproducibility if small and useful.

## Optional DVC Artifact Tracking

If you want reproducible model versioning across Kaggle runs, use DVC after downloading outputs locally:

1. dvc add models/autoencoder.pt
2. dvc add models/isolation_forest.joblib
3. dvc add artifacts/eval_report.json
4. git add models/*.dvc artifacts/*.dvc .gitignore
5. git commit -m "Track trained artifacts with DVC"
6. dvc push

## GPU Validation

Inside Kaggle notebook, you can verify GPU availability with:

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

The script uses --device auto and will select cuda when available.
