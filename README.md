# Age-invariant face retrieval based on Hybrid Metric Learning Framework (HMLF)

This repository is based on the article Age-invariant face retrieval based on Hybrid Metric Learning Framework (HMLF), and includes its experiments for cross-age face retrieval on 5 different datasets.  
The main entry scripts all start with **online_train_**, where each script corresponds to one dataset.

---

## 1. Main Entry Scripts (5 Datasets)

- AgeDB: [dirtorch/online_train_agedb.py](dirtorch/online_train_agedb.py)  
- CACD2000: [dirtorch/online_train_cacd.py](dirtorch/online_train_cacd.py)  
- FG-NET: [dirtorch/online_train_fgnet.py](dirtorch/online_train_fgnet.py)  
- IMDB-clean-1024: [dirtorch/online_train_imdb.py](dirtorch/online_train_imdb.py)  
- MORPH: [dirtorch/online_train_morph.py](dirtorch/online_train_morph.py)

These scripts typically do:
- Load the corresponding dataset via `datasets.create(...)`
- Train and evaluate the model
- Report metrics such as `mAP`, `P@1`, etc., and optionally save Top-K retrieval visualizations

---

## 2. Dependencies

- Python 3 (recommended 3.8+)
- PyTorch + torchvision
- Common packages: numpy, pillow, tqdm, matplotlib, scikit-learn, pandas
- `timm` (Swin-Transformer code relies on it)

Example (pip):
```bash
pip install numpy pillow tqdm matplotlib scikit-learn pandas
pip install timm
# Install PyTorch/torchvision according to your CUDA/CPU environment
```

---

## 3. Required Environment Variables (Important)

Before running, you must set these environment variables (otherwise you will hit `KeyError`):

- `DIR_ROOT`: repository root directory (used to load pretrained weights stored inside the repo)
- `DB_ROOT`: datasets root directory (used to locate images and pickle annotations)

### Windows (PowerShell, effective only in current terminal)
```powershell
$env:DIR_ROOT="D:\path\to\age-invariant-face-retrieval-based-on-HMLF"
$env:DB_ROOT="D:\path\to\datasets"
```

### Windows (cmd, effective only in current terminal)
```bat
set DIR_ROOT=D:\path\to\age-invariant-face-retrieval-based-on-HMLF
set DB_ROOT=D:\path\to\datasets
```

### Linux/macOS
```bash
export DIR_ROOT=/path/to/age-invariant-face-retrieval-based-on-HMLF
export DB_ROOT=/path/to/datasets
```

---

## 4. Dataset Preparation (DB_ROOT Directory Layout)

Dataset classes read `DB_ROOT` at import time (e.g., [dirtorch/datasets/agedb.py](dirtorch/datasets/agedb.py)), so make sure the required images and pickle files exist under `DB_ROOT`.

### 4.1 AgeDB
- Hugging Face：[caojingtian1216/AgeDB · Datasets at Hugging Face](https://huggingface.co/datasets/caojingtian1216/AgeDB)
- Definition: [dirtorch/datasets/agedb.py](dirtorch/datasets/agedb.py)  
- Images: `DB_ROOT/AgeDB/archive/jpg/*.jpg`  
- Required pickle files:
  - `DB_ROOT/AgeDB/gnd_agedb.pkl`
  - `DB_ROOT/AgeDB/agedb_train.pkl`
  - `DB_ROOT/AgeDB/agedb_trainvalid.pkl`

### 4.2 CACD
- Hugging Face：[caojingtian1216/CACD · Datasets at Hugging Face](https://huggingface.co/datasets/caojingtian1216/CACD)
- Definition: [dirtorch/datasets/CADA2000.py](dirtorch/datasets/CADA2000.py)  
- Images: `DB_ROOT/CACD/data/jpg/*.jpg`  
- Required pickle files:
  - `DB_ROOT/CACD/data/gnd_cada1.pkl`
  - `DB_ROOT/CACD/data/gnd_cada2.pkl`
  - `DB_ROOT/CACD/data/gnd_cada3.pkl`
  - `DB_ROOT/CACD/data/cada_train.pkl`
  - `DB_ROOT/CACD/data/cada_trainvalid.pkl`

### 4.3 FG-NET
- Hugging Face：[caojingtian1216/FGNET · Datasets at Hugging Face](https://huggingface.co/datasets/caojingtian1216/FGNET)
- Definition: [dirtorch/datasets/fgnet.py](dirtorch/datasets/fgnet.py)  
- Images: `DB_ROOT/FGNET/jpg/*.jpg`  
- Required pickle files:
  - `DB_ROOT/FGNET/gnd_fgnet.pkl`
  - `DB_ROOT/FGNET/fgnet_train.pkl`
  - `DB_ROOT/FGNET/fgnet_trainvalid.pkl`

### 4.4 IMDB-clean-1024
- Hugging Face：[caojingtian1216/imdb-clean-1024 · Datasets at Hugging Face](https://huggingface.co/datasets/caojingtian1216/imdb-clean-1024)
- Definition: [dirtorch/datasets/imdb.py](dirtorch/datasets/imdb.py)  
- Images: `DB_ROOT/imdb-clean-1024/jpg/**`  
- Required pickle files:
  - `DB_ROOT/imdb-clean-1024/gnd_imdb.pkl`
  - `DB_ROOT/imdb-clean-1024/imdb_train.pkl`
  - `DB_ROOT/imdb-clean-1024/imdb_trainvalid.pkl`

### 4.5 MORPH
- Hugging Face：[caojingtian1216/morph · Datasets at Hugging Face](https://huggingface.co/datasets/caojingtian1216/morph)
- Definition: [dirtorch/datasets/morph.py](dirtorch/datasets/morph.py)  
- Images: `DB_ROOT/morph/data/jpg/*` (extension/case depends on your dataset)  
- Required pickle files:
  - `DB_ROOT/morph/data/gnd_morph.pkl`
  - `DB_ROOT/morph/data/morph_train.pkl`
  - `DB_ROOT/morph/data/morph_trainvalid.pkl`

---

## 5. How to Run (Recommended: python -m)

Run from the repository root so that `dirtorch` can be imported correctly:

```bash
python -m dirtorch.online_train_agedb  --gpu 0 --threads 4
python -m dirtorch.online_train_cacd   --gpu 0 --threads 4
python -m dirtorch.online_train_fgnet  --gpu 0 --threads 4
python -m dirtorch.online_train_imdb   --gpu 0 --threads 4
python -m dirtorch.online_train_morph  --gpu 0 --threads 4
```

CPU example:
```bash
python -m dirtorch.online_train_agedb --gpu -1
```

---

## 6. Input Size / Normalization

Default input size and normalization are defined in [dirtorch/config.py](dirtorch/config.py):
- `cfg.INPUT_SIZE` (currently defaults to 112)
- `cfg.MEAN` / `cfg.STD`

The data loader also uses this configuration for Resize/RandomCrop, see [dirtorch/utils/pytorch_loader.py](dirtorch/utils/pytorch_loader.py).

---

## 7. Pretrained Weights (Used by Default in Scripts)

Common weight files (stored in Hugging Face):
- InceptionResnetV1 (vggface2): [caojingtian1216/FaceNet · Hugging Face](https://huggingface.co/caojingtian1216/FaceNet)
- IResNet50 backbone: [caojingtian1216/IResnet · Hugging Face](https://huggingface.co/caojingtian1216/IResnet)
- MobileFaceNet: [caojingtian1216/MobileFaceNet · Hugging Face](https://huggingface.co/caojingtian1216/MobileFaceNet)
- Swin Transformer: [caojingtian1216/Swin-Transformer · Hugging Face](https://huggingface.co/caojingtian1216/Swin-Transformer)

Different online_train_* scripts use different default backbones.  
If you want to switch backbones, you typically need to modify the `net = ...` block inside the corresponding script (some scripts keep a `--checkpoint` argument but may not actually use it).

---

## 8. Important Notes (Read First)

- Several scripts include `cleanup_and_save(...)`:
  - The output directory is often hard-coded to `/kaggle/working`
  - It may delete everything in the output directory except `.pt` / `.jpg`, and then call `os._exit(0)` to force-quit  
  **When running locally, always change the output directory to a dedicated empty folder to avoid accidental deletion.**
- If you see `KeyError: 'DB_ROOT'` or `KeyError: 'DIR_ROOT'`, your environment variables are not set (or not set in the current terminal/session).
- AgeDB uses Swin by default: if you get `ModuleNotFoundError: timm`, install `timm`.

---

## 9. License

MIT License. See [License](https://github.com/caojingtian1216/age-invariant-face-retrieval-based-on-HMLF/blob/main/LICENSE).
