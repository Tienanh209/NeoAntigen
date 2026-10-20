## Workflow: local (VS Code) → GitHub → Colab, data on Drive

- **Code**: edit locally, `git push` to `Tienanh209/NeoAntigen`. Colab clones/pulls it to the VM disk (`/content/NeoAntigen`). Do not edit code on Colab, and do not clone the repo into Drive.
- **Data and checkpoints**: Drive, `/content/drive/MyDrive/s1156052/NeoAntigen/{data,ckpt}`; never in git. Before training, `config.stage_data()` copies data to `/content/data`.
- **Paths**: only through `src/config.py` (`DATA_DIR`, `CKPT_DIR`, `WORK_DIR`); it detects Colab vs local. Never hardcode absolute paths elsewhere.
- **Logic lives in `src/*.py`**; notebooks are thin launchers (setup, parameters, calls, printing). Strip notebook outputs before committing.
- **Dataset versions**: raw CEDAR is saved as `data/cedar_raw_YYYYMMDD.csv` (CEDAR changes over time) and each build writes `data/meta.json` with its parameters.
- **GitHub token**: Colab secret `GH_TOKEN`, never in notebooks or commits.
- **Colab installs**: never `pip install torch/torchvision/torchaudio/nvidia-*`; use
  `grep -viE '^(torch|torchvision|torchaudio|nvidia-)' requirements.txt > req_colab.txt`
- **Training**: save a checkpoint to Drive every epoch and make it resumable (epoch, model, optimizer in `last.pt`). LoRA: save the adapter only.
- **Never commit** data, weights (`*.pt`, `*.pth`, `*.ckpt`), `.env`, or tokens.
