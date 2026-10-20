## Workflow: local (VS Code) → GitHub → Colab, data on Drive

- **Code**: edit locally, `git push` to `Tienanh209/NeoAntigen`. Colab clones/pulls it to the VM disk (`/content/NeoAntigen`). Do not edit code on Colab, and do not clone the repo into Drive.
- **Data and checkpoints**: Drive, `/content/drive/MyDrive/s1156052/NeoAntigen/{data,ckpt}`; never in git. Before training, `config.stage_data()` copies data to `/content/data`.
- **Paths**: only through `src/config.py` (`DATA_DIR`, `CKPT_DIR`, `WORK_DIR`); it detects Colab vs local. Never hardcode absolute paths elsewhere.
- **Logic lives in `src/*.py`**; notebooks are thin launchers (setup, parameters, calls, printing). Strip notebook outputs before committing.
- **Dataset versions**: raw CEDAR is saved as `data/cedar_raw_YYYYMMDD.csv` (CEDAR changes over time) and each build writes `data/meta.json` with its parameters.
- **GitHub token**: Colab secret `GH_TOKEN`, never in notebooks or commits.
- **Colab installs**: never `pip install torch/torchvision/torchaudio/nvidia-*`, and never `numpy`/`scipy` either — Colab ships numpy 2.x with a torch built against it, so the local `numpy<2` pin silently breaks the preinstalled torch. Use
  `grep -viE '^(torch|torchvision|torchaudio|nvidia-|numpy|scipy)' requirements.txt > req_colab.txt`
- **Experiments**: `src/runner.py` is the entry point; one notebook block per experiment. Every run is committed to `CKPT_DIR/results/` keyed by a config hash, so re-running a block skips finished work and a crash never loses earlier results. Tables and ROC figures are rebuilt from the stored per-sample predictions, never from cached metrics.
- **Embeddings**: PLMs are frozen and embedded once into `DATA_DIR/emb/{build_id}/{plm}/`; experiments read that cache and never load a PLM.
- **Training**: save a checkpoint to Drive every epoch and make it resumable (epoch, model, optimizer in `last.pt`). LoRA: save the adapter only.
- **Never commit** data, weights (`*.pt`, `*.pth`, `*.ckpt`), `.env`, or tokens.
