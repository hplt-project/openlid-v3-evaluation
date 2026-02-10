# OpenLID-v3: Precision-Driven Language Identification Tool

This repository contains data and code to reproduce evaluations, described in the paper "OpenLID-v3: Precision-Driven Language Identification Tool", accepted to [VarDial 2026](https://sites.google.com/view/vardial-2026/) workshop, collocated with [EACL 2026](https://2026.eacl.org/).

For training the OpenLID-v3 model and running it for inference, refer to [its repository](https://github.com/hplt-project/openlid).

## Evaluate OpenLID-v3

Get the model

```shell
wget https://zenodo.org/records/17601701/files/openlid-v3.bin
```

`cd evaluation/`

!beware of hardcoded paths. The scripts are meant to be run from `evaluation/`

### FLORES+ (all languages)

#### Get data

`scripts/download_flores_plus.py`

#### Predict

```shell
python3 scripts/fasttext_predictions.py \
    --dataset flores \
    --split devtest \
    --model retrained \
    --model-path <path to the model> \
    --enable-preprocessing \
    --out_path <where to save the result>.jsonl
```

For this and other benchmarks, add `--threshold` to use softmax thresholding at 0.5 and `--ensemble_with_glotlid_k 1` to ensemble with GlotLID

#### Evaluate

```shell
python3 scripts/evaluate.py \
  <out_path from the previous step>.jsonl \
  --model retrained \
  --languages-file language-lists/openlid-flores-glotlid-udhr.txt \
  --dataset flores > <path to the result>.json
```

### UDHR (all languages)

#### Get data

`scripts/download_udhr.py`

#### Predict

```shell
python3 scripts/fasttext_predictions.py \
    --dataset udhr \
    --model retrained \
    --model-path <path to the model> \
    --enable-preprocessing \
    --out_path <where to save the result>.jsonl
```

#### Evaluate

```shell
python3 scripts/evaluate.py \
  <out_path from the previous step>.jsonl \
  --model retrained \
  --languages-file language-lists/openlid-flores-glotlid-udhr.txt \
  --dataset udhr > <path to the result>.json
```

### FastSpell (many languages, noisy web data)

#### Get data

```shell
cd ../new_benchmarks_creation/benchmarks/
git submodule init
git submodule update
cd ..
python3 create_fastspell_dataset.py
```

#### Predict

```shell
python3 scripts/fasttext_predictions.py \
    --dataset fastspell \
    --model retrained \
    --model-path <path to the model> \
    --enable-preprocessing \
    --out_path <where to save the result>.jsonl
```

#### Evaluate

For FastSpell and the rest of benchmarks, it is the same, as for UDHR (use `--dataset udhr` to define the format, a corresponding `--languages-file` and pass the correct path to predictions)

### HPLT 3.0 (many languages, noisy web data)

#### Get data

```shell
cd ../new_benchmarks_creation/release3_inspection/
git submodule init
git submodule update
cd ..
python3 create_hplt_dataset.py
```

#### Predict

```shell
cd ../../evaluation
python3 scripts/fasttext_predictions.py \
    --dataset hplt \
    --model retrained \
    --model-path <path to the model> \
    --enable-preprocessing \
    --out_path <where to save the result>.jsonl
```

### Twitter (BCS)

It's multilabel, so see [the SLIDE repository](https://github.com/ltgoslo/slide) 

### ParlaSent (BCS)

#### Get data

```shell
cd ../
python3 create_hbs_datasets.py --dataset parlasent
```

#### Predict

```shell
cd ../evaluation
python3 scripts/fasttext_predictions.py \
    --dataset parlasent \
    --model retrained \
    --model-path <path to the model> \
    --enable-preprocessing \
    --out_path <where to save the result>.jsonl
```

### ITDI (languages of Italy)

#### Get data

```shell
cd ../new_benchmarks_creation/ITDI_2022/
git submodule init
git submodule update
cd ..
python3 create_ligurian_dataset.py
```

#### Predict

```shell
cd ../evaluation
python3 scripts/fasttext_predictions.py \
    --dataset ITDI_2022 \
    --model retrained \
    --model-path <path to the model> \
    --enable-preprocessing \
    --out_path <where to save the result>.jsonl
```

### SLIDE, NordicDSL (Scandinavian)

see [the SLIDE repository](https://github.com/ltgoslo/slide)

---------------------------------------------------------------------------------------------

<sub><sup>This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101070350 and from UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number 10052546].
The contents of this publication are the sole responsibility of the HPLT consortium and do not necessarily reflect the opinion of the European Union.</sup></sub>