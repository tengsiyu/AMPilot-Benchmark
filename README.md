# AMPilot-Benchmark

An open benchmark for motion planning with large language models in open-pit mines.

AMPilot-Benchmark provides instruction-style datasets, prediction files, classical learning baselines, and evaluation code for trajectory planning in open-pit mining scenarios. The benchmark is designed to support reproducible comparison between LLM-based planners and conventional neural trajectory prediction models.

## Overview

Open-pit mines contain large-scale, structured, and safety-critical traffic scenes. A planning model must understand mining truck states, local traffic context, maneuver intention, and future trajectory evolution. AMPilot-Benchmark evaluates this capability through four test scenarios and a unified metric suite.

This repository contains:

- instruction-following data for mining-traffic motion planning;
- LLM prediction outputs with Chain-of-Thought and non-Chain-of-Thought settings;
- classical trajectory prediction baselines;
- scenario-level benchmark results in JSON and XLSX formats;
- an evaluation script for recomputing AMPilot-Metrix scores.

## News

- 2026-05-19: Initial public release of AMPilot-Benchmark.

## Repository Structure

```text
AMPilot-Benchmark/
|-- AMPilot-Dataset/
|   |-- LC-LLM_CoT.json
|   |-- LC-LLM_noCoT.json
|   |-- MiningCoT.json
|   |-- Mining_noCoT.json
|   `-- TestDataset/
|       |-- Scenario0/
|       |-- Scenario1/
|       |-- Scenario2/
|       |-- Scenario3/
|       |-- Scenario0Comparison/
|       |-- Scenario1Comparison/
|       |-- Scenario2Comparison/
|       `-- Scenario3Comparison/
|-- AMPilot-Metrix/
|   `-- AMPilot_metrix.py
|-- ClassicialModels/
|   |-- train_all_models_AMPolit.py
|   |-- test_all_model_AMPilot.py
|   |-- GRU_LocalModel.py
|   |-- LSTM_LocalCoor.py
|   |-- RNN_LocalCoor.py
|   |-- Transform_LocalCoor.py
|   |-- ST_GCN_Local.py
|   |-- MMTP_prediction_Local.py
|   |-- KANLSTM_Local.py
|   `-- Bi_IFNet_local_unified.py
|-- .gitattributes
|-- .gitignore
`-- README.md
```

Note: the directory name `ClassicialModels` is kept as released in this repository.

## Dataset

`AMPilot-Dataset/` contains the benchmark data and released predictions.

The top-level JSON files provide instruction-style data under different prompting or model families:

- `MiningCoT.json`
- `Mining_noCoT.json`
- `LC-LLM_CoT.json`
- `LC-LLM_noCoT.json`

`AMPilot-Dataset/TestDataset/` contains four scenario folders:

- `Scenario0`
- `Scenario1`
- `Scenario2`
- `Scenario3`

Each scenario folder contains LLM prediction files and classical-model prediction files. The corresponding `Scenario*Comparison/` folders contain precomputed AMPilot-Metrix results in JSON and XLSX formats.

## Benchmark Tasks

AMPilot-Benchmark evaluates future trajectory planning for open-pit mining vehicles. Each evaluated sample contains a predicted trajectory and a reference trajectory over a fixed future horizon.

The benchmark also supports LLM-specific evaluation of:

- trajectory prediction quality;
- driving or maneuver intention prediction;
- mining-truck load-state reasoning, when available.

## Baselines

The benchmark includes released predictions from LLM planners and classical trajectory prediction models.

LLM-style methods include:

- `MiningCoT7B`
- `MiningCoT13B`
- `MiningNoCoT7B`
- `MiningNoCoT13B`
- `LC-LLM_CoT7B`
- `LC-LLM_CoT13B`
- `LC-LLMNoCoT7B`
- `LC-LLMNoCoT13B`

Classical baselines include:

- GRU
- LSTM
- RNN
- Transformer
- ST-GCN
- MMTP
- KAN-LSTM
- Bi-IFNet

## AMPilot-Metrix

The evaluation code is located in `AMPilot-Metrix/AMPilot_metrix.py`.

The main metrics include:

- `RMSE`: root mean squared trajectory error;
- `ADE`: average displacement error;
- `FDE`: final displacement error;
- `AHE`: average heading error;
- `FHE`: final heading error;
- per-step x-axis and y-axis RMSE;
- intention prediction precision, recall, and F1 for LLM planners;
- load-state evaluation for supported MiningCoT models.

Lower values are better for `RMSE`, `ADE`, `FDE`, `AHE`, and `FHE`.

## Getting Started

### Clone the Repository

This repository uses Git LFS for large JSON and XLSX files. Install Git LFS before cloning or pulling the data.

```bash
git lfs install
git clone https://github.com/tengsiyu/AMPilot-Benchmark.git
cd AMPilot-Benchmark
git lfs pull
```

### Environment

The metric script requires Python and the following packages:

```bash
pip install numpy openpyxl
```

The classical baseline scripts additionally use PyTorch, scikit-learn, pandas, and matplotlib:

```bash
pip install torch scikit-learn pandas matplotlib
```

Install the PyTorch build that matches your CUDA environment if you plan to train or test the neural baselines on GPU.

## Evaluation

Run AMPilot-Metrix on a scenario folder:

```bash
python AMPilot-Metrix/AMPilot_metrix.py \
  --llm-dir AMPilot-Dataset/TestDataset/Scenario2 \
  --classical-dir AMPilot-Dataset/TestDataset/Scenario2 \
  --output-dir AMPilot-Dataset/TestDataset/Scenario2Comparison \
  --scenario Scenario2
```

Evaluate only LLM predictions:

```bash
python AMPilot-Metrix/AMPilot_metrix.py \
  --llm-dir AMPilot-Dataset/TestDataset/Scenario2 \
  --output-dir AMPilot-Dataset/TestDataset/Scenario2Comparison \
  --scenario Scenario2 \
  --skip-classical
```

Evaluate only classical predictions:

```bash
python AMPilot-Metrix/AMPilot_metrix.py \
  --classical-dir AMPilot-Dataset/TestDataset/Scenario2 \
  --output-dir AMPilot-Dataset/TestDataset/Scenario2Comparison \
  --scenario Scenario2 \
  --skip-llm
```

The script writes:

- `scenario*_ampilot_metrix.json`
- `scenario*_ampilot_metrix.xlsx`

## Results

The following table reports the best released method in each scenario according to RMSE. Values are computed from the released `Scenario*Comparison/scenario*_ampilot_metrix.json` files.

| Scenario | Best Method | Group | Samples | RMSE | ADE | FDE | AHE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scenario0 | MiningCoT13B | LLM | 5000 | 7.6960 | 2.4866 | 5.2063 | 0.0181 |
| Scenario1 | MiningCoT13B | LLM | 5000 | 16.7453 | 11.5707 | 23.5061 | 0.1477 |
| Scenario2 | MiningCoT13B | LLM | 5000 | 14.3216 | 10.1609 | 21.4216 | 0.0774 |
| Scenario3 | MiningCoT13B | LLM | 5000 | 15.0454 | 10.2559 | 20.4606 | 0.0956 |

Full results are available in:

- `AMPilot-Dataset/TestDataset/Scenario0Comparison/`
- `AMPilot-Dataset/TestDataset/Scenario1Comparison/`
- `AMPilot-Dataset/TestDataset/Scenario2Comparison/`
- `AMPilot-Dataset/TestDataset/Scenario3Comparison/`

## Classical Models

Training and testing utilities are provided in `ClassicialModels/`.

Train all released classical models:

```bash
python ClassicialModels/train_all_models_AMPolit.py \
  --train-file AMPilot-Dataset/MiningCoT.json \
  --pred-dir AMPilot-Dataset/TestDataset/Scenario0 \
  --epochs 300 \
  --batch-size 1024
```

Test classical models:

```bash
python ClassicialModels/test_all_model_AMPilot.py \
  --train AMPilot-Dataset/MiningCoT.json \
  --pred-dir AMPilot-Dataset/TestDataset/Scenario0 \
  --allow-cpu
```

The scripts expose additional options for device selection, batch size, learning rate, random seed, and CPU fallback.

## Add Your Own Method

To evaluate a new planner:

1. Save prediction files into the target scenario folder under `AMPilot-Dataset/TestDataset/Scenario*/`.
2. Follow the JSON structure used by the released prediction files.
3. Run `AMPilot-Metrix/AMPilot_metrix.py` with the target scenario.
4. Compare the generated JSON/XLSX results with the released benchmark results.

For LLM planners, the evaluator extracts trajectories from the `Trajectory` field or from text outputs containing a trajectory block. For classical models, prediction files should use `label` or `Label` and `prediction` or `Prediction` fields.

## Acknowledgement

We thank the open-source research community for tools and benchmarks that support reproducible autonomous driving and motion-planning research.

## Citation

If you find AMPilot-Benchmark useful for your research, please cite our work. The BibTeX entry will be updated after the associated paper is publicly available.

```bibtex
@article{ampilotbenchmark2026,
  title  = {AMPilot-Benchmark: An Open Benchmark for Motion Planning with Large Language Models in Open-Pit Mines},
  author = {TBD},
  journal = {TBD},
  year   = {2026}
}
```

## License

The license for this repository will be clarified in a future update. Unless a license file is added, please contact the authors before using the dataset or code for commercial purposes.
