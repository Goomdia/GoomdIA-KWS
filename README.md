# GoomdIA KWS research code

Research code accompanying *A Spoken Keyword Corpus and Speaker-Independent Benchmarks for Mooré*, by Palamanga Onadja, Andie Samadoulougou, Arnaud Yarga, Benoit Kiema, Rabiatou Djebré and Blaise Madiega (GoomdIA, Ouagadougou, Burkina Faso).

This repository contains the curation, training, evaluation and analysis code (MIT licence) and the aggregate result tables of the article. The compact models (PyTorch checkpoints and TensorFlow Lite exports, CC BY 4.0) are attached to the release and archived at https://doi.org/10.5281/zenodo.24029301. The audio recordings, per-clip metadata, dataset splits and test lists are voice data of identifiable persons and are not publicly distributed: GoomdIA provides them to researchers on request under a data-use agreement (contact@goomdia.com).

## Contents

- `analysis/`: corpus analysis, aggregation, plotting and table-generation scripts.
- `experiments/`: training, evaluation, model, deployment and orchestration code.
- `experiments/requirements.txt`: recorded training environment.
- `data/`: aggregate result tables (accuracies, confusion matrices, deployment, streaming); no per-clip data.
- `LICENSE`: MIT licence for the code.
- `LICENSE-CC-BY-4.0.txt`: CC BY 4.0 licence for the result tables in `data/` and for the compact models.
- `SHA256SUMS`: checksums of the code package.

## Running the code

Install the dependencies in `experiments/requirements.txt` with the appropriate PyTorch CUDA package index for the `+cu121` builds. Some deployment and model scripts import additional optional dependencies.

The scripts require local data and configuration paths. The corpus, splits and test lists are not included; retraining or re-evaluating requires the data obtained on request. Configure `GOOMDIA_CACHE` and the command-line paths for data you are authorised to use. Earlier analysis stages can overwrite tables; inspect the scripts before running them.

The code licence grants no right to the corpus, which is shared only under a data-use agreement.

Contact: contact@goomdia.com.
