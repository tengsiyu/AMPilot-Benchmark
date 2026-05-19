# AMPilot-Benchmark

This is the official repository of **AMPilot: An Open Benchmark for End-to-End Autonomous Driving with Mixed Traffic in Open-pit Mines**

AMPilot is an open benchmark for motion planning with large language models in open-pit mines.

AMPilot-Benchmark provides instruction-style datasets, prediction files, classical learning baselines, and evaluation code for trajectory planning in mixed traffic of open-pit mining scenarios. The benchmark is designed to support reproducible comparison between LLM-based planners and conventional neural trajectory prediction models.

## Overview

Open-pit mines contain large-scale, structured, and safety-critical traffic scenes. A planning model must understand mining truck states, local traffic context, maneuver intention, and future trajectory evolution. AMPilot-Benchmark evaluates this capability through four test scenarios and a unified metric suite.

<p align="center">
  <img src="Assets/HardwarePlarform.png" alt="AMPilot data collection hardware platform" width="48%">
  <img src="Assets/scenario.png" alt="AMPilot benchmark scenarios" width="48%">
</p>

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

The following example shows one MiningCoT frame, including the surrounding context, instruction prompt, and generated output.

<p align="center">
  <img src="Assets/F_Instructions.png" alt="MiningCoT instruction example" width="78%">
</p>


<p align="center">
  <img src="Assets/F_surroundings.png" alt="MiningCoT surrounding context example" width="78%">
</p>


<p align="center">
  <img src="Assets/F_outputs.png" alt="MiningCoT output example" width="78%">
</p>

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
```

### Environment

Create an environment using Python 3.10:

```bash
conda env create -f AMPilot_environment.yml -n ampilot
source activate AMPilot_environment
```

Follow [the official documentation](https://llama2-accessory.readthedocs.io/) to set up the LLaMA2-Accessory environment.

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

The following tables list all released experimental results from `AMPilot-Dataset/TestDataset/Scenario*Comparison/scenario*_ampilot_metrix.json`. Lower is better for RMSE, ADE, AHE, FDE, and FHE.

### Trajectory Prediction Results

<details open>
<summary>Scenario0</summary>

<p align="center">
  <img src="Assets/S0_MSE_X_axis.png" alt="Scenario0 MSE on x-axis" width="48%">
  <img src="Assets/S0_MSE_Y_axis.png" alt="Scenario0 MSE on y-axis" width="48%">
</p>

| Group | Model | RMSE | ADE | AHE | FDE | FHE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Classical | KANLSTM | 12.9371 | 7.5791 | 0.2428 | 14.2510 | 0.2208 |
| Classical | BiIFNet | 13.0615 | 8.0592 | 0.3376 | 14.6477 | 0.3188 |
| Classical | MMTP | 14.0710 | 8.2521 | 0.4564 | 14.1539 | 0.4582 |
| Classical | ST_GCN | 14.0850 | 8.3613 | 0.2413 | 15.8083 | 0.1427 |
| Classical | GRU | 16.7187 | 10.3809 | 0.1758 | 19.0631 | 0.1863 |
| Classical | LSTM | 16.8753 | 11.1061 | 0.3791 | 19.5961 | 0.3207 |
| Classical | Transformer | 18.3410 | 12.6249 | 0.3120 | 21.8309 | 0.2729 |
| Classical | RNN | 18.4796 | 12.5149 | 0.3132 | 22.4259 | 0.3584 |
| LLM | MiningCoT13B | 7.6960 | 2.4866 | 0.0181 | 5.2063 | 0.0285 |
| LLM | MiningNoCoT13B | 8.1294 | 2.5720 | 0.0182 | 5.5140 | 0.0285 |
| LLM | MiningCoT7B | 10.3717 | 3.5713 | 0.0234 | 7.5940 | 0.0364 |
| LLM | LC-LLM_CoT13B | 10.6050 | 3.8599 | 0.0241 | 8.2185 | 0.0376 |
| LLM | LC-LLMNoCoT13B | 10.6815 | 3.5769 | 0.0202 | 7.5826 | 0.0324 |
| LLM | MiningNoCoT7B | 10.8559 | 3.7695 | 0.0236 | 7.6305 | 0.0362 |
| LLM | LC-LLMNoCoT7B | 11.4914 | 4.1927 | 0.0245 | 8.8224 | 0.0369 |
| LLM | LC-LLM_CoT7B | 12.3996 | 4.7235 | 0.0270 | 10.0001 | 0.0425 |

</details>

<details open>
<summary>Scenario1</summary>

<p align="center">
  <img src="Assets/S1_MSE_X_axis.png" alt="Scenario1 MSE on x-axis" width="48%">
  <img src="Assets/S1_MSE_Y_axis.png" alt="Scenario1 MSE on y-axis" width="48%">
</p>

| Group | Model | RMSE | ADE | AHE | FDE | FHE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Classical | Transformer | 20.3163 | 15.5166 | 0.2345 | 27.8518 | 0.2815 |
| Classical | RNN | 20.7423 | 16.6083 | 0.2116 | 29.2732 | 0.2901 |
| Classical | KANLSTM | 21.2497 | 15.8423 | 0.2077 | 28.9712 | 0.2873 |
| Classical | MMTP | 22.9907 | 15.5730 | 0.2944 | 29.2336 | 0.4747 |
| Classical | BiIFNet | 23.0930 | 15.8838 | 0.2094 | 28.5653 | 0.3008 |
| Classical | GRU | 23.4004 | 16.9893 | 0.2448 | 31.0227 | 0.3226 |
| Classical | ST_GCN | 23.6716 | 15.5784 | 0.2063 | 27.8526 | 0.2786 |
| Classical | LSTM | 24.1520 | 19.5216 | 0.2031 | 34.0848 | 0.2773 |
| LLM | MiningCoT13B | 16.7453 | 11.5707 | 0.1477 | 23.5061 | 0.2295 |
| LLM | LC-LLMNoCoT7B | 16.9187 | 11.8958 | 0.1358 | 24.2533 | 0.2129 |
| LLM | MiningNoCoT13B | 17.0733 | 12.0257 | 0.1421 | 24.7465 | 0.2208 |
| LLM | MiningCoT7B | 17.4657 | 12.6912 | 0.1231 | 24.5863 | 0.1931 |
| LLM | MiningNoCoT7B | 17.5311 | 12.7698 | 0.1272 | 25.1851 | 0.2049 |
| LLM | LC-LLM_CoT7B | 18.0036 | 12.0749 | 0.1386 | 23.9815 | 0.2168 |
| LLM | LC-LLM_CoT13B | 18.2666 | 13.4161 | 0.1410 | 25.6550 | 0.2200 |
| LLM | LC-LLMNoCoT13B | 18.3359 | 13.4121 | 0.1380 | 26.3499 | 0.2122 |

</details>

<details open>
<summary>Scenario2</summary>

<p align="center">
  <img src="Assets/S2_MSE_X_axis.png" alt="Scenario2 MSE on x-axis" width="48%">
  <img src="Assets/S2_MSE_Y_axis.png" alt="Scenario2 MSE on y-axis" width="48%">
</p>

| Group | Model | RMSE | ADE | AHE | FDE | FHE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Classical | GRU | 19.7205 | 14.1094 | 0.0863 | 27.1983 | 0.1109 |
| Classical | RNN | 24.7095 | 18.8936 | 0.0991 | 35.0796 | 0.1453 |
| Classical | LSTM | 26.7257 | 21.0131 | 0.0881 | 38.3455 | 0.1086 |
| Classical | ST_GCN | 39.5851 | 29.7885 | 0.1618 | 56.2450 | 0.2307 |
| Classical | MMTP | 39.8446 | 26.1879 | 0.3876 | 53.7663 | 0.6059 |
| Classical | KANLSTM | 42.2598 | 32.8544 | 0.1519 | 62.1642 | 0.2098 |
| Classical | Transformer | 42.9811 | 36.6544 | 0.4219 | 66.2537 | 0.2964 |
| Classical | BiIFNet | 45.1793 | 33.9248 | 0.2776 | 64.3436 | 0.3535 |
| LLM | MiningCoT13B | 14.3216 | 10.1609 | 0.0774 | 21.4216 | 0.0955 |
| LLM | LC-LLM_CoT13B | 14.5493 | 10.1132 | 0.0862 | 20.7233 | 0.1095 |
| LLM | MiningNoCoT7B | 15.5919 | 10.6168 | 0.0710 | 23.6064 | 0.0911 |
| LLM | LC-LLMNoCoT13B | 16.3889 | 11.3319 | 0.0975 | 23.1925 | 0.1172 |
| LLM | MiningNoCoT13B | 18.4416 | 12.5824 | 0.0793 | 28.3387 | 0.1004 |
| LLM | MiningCoT7B | 21.7481 | 14.2354 | 0.0784 | 30.2092 | 0.1019 |
| LLM | LC-LLM_CoT7B | 23.8766 | 14.0866 | 0.0810 | 26.1955 | 0.1036 |
| LLM | LC-LLMNoCoT7B | 25.2423 | 15.2621 | 0.0798 | 29.9358 | 0.1031 |

</details>

<details open>
<summary>Scenario3</summary>

<p align="center">
  <img src="Assets/S3_MSE_X_axis.png" alt="Scenario3 MSE on x-axis" width="48%">
  <img src="Assets/S3_MSE_Y_axis.png" alt="Scenario3 MSE on y-axis" width="48%">
</p>

| Group | Model | RMSE | ADE | AHE | FDE | FHE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Classical | KANLSTM | 19.9583 | 14.4090 | 0.2432 | 26.8944 | 0.3575 |
| Classical | RNN | 20.6539 | 16.0468 | 0.2191 | 28.4656 | 0.3030 |
| Classical | BiIFNet | 21.4287 | 14.7692 | 0.2853 | 27.4020 | 0.4051 |
| Classical | ST_GCN | 22.5225 | 14.9137 | 0.2243 | 26.7522 | 0.3035 |
| Classical | Transformer | 23.2188 | 16.5462 | 0.2299 | 28.6754 | 0.2952 |
| Classical | MMTP | 23.3976 | 15.6536 | 0.3366 | 30.0986 | 0.5086 |
| Classical | LSTM | 23.5680 | 17.9601 | 0.2451 | 32.1027 | 0.3306 |
| Classical | GRU | 24.8602 | 16.9418 | 0.2438 | 31.0480 | 0.3367 |
| LLM | MiningCoT13B | 15.0454 | 10.2559 | 0.0956 | 20.4606 | 0.1691 |
| LLM | LC-LLMNoCoT7B | 15.5654 | 10.8082 | 0.1557 | 21.7317 | 0.2423 |
| LLM | MiningNoCoT13B | 16.0939 | 11.5921 | 0.1508 | 22.2805 | 0.2320 |
| LLM | LC-LLM_CoT7B | 16.3828 | 11.0503 | 0.1525 | 21.8677 | 0.2369 |
| LLM | MiningNoCoT7B | 16.4857 | 11.7902 | 0.1481 | 22.9721 | 0.2339 |
| LLM | MiningCoT7B | 17.1317 | 12.0141 | 0.1490 | 23.2989 | 0.2312 |
| LLM | LC-LLMNoCoT13B | 17.4117 | 12.3432 | 0.1554 | 24.1833 | 0.2376 |
| LLM | LC-LLM_CoT13B | 17.5228 | 12.7537 | 0.1516 | 24.6373 | 0.2348 |

</details>

### LLM Intention and Load-State Results

The intention columns report macro-averaged precision, recall, and F1. Load-state accuracy is reported when the model output contains a valid load-state decision.

<details open>
<summary>Scenario1</summary>

| Model | Intention Macro-P | Intention Macro-R | Intention Macro-F1 | Load-State Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| MiningCoT13B | 0.9790 | 0.9653 | 0.9718 | 1.0000 |
| MiningNoCoT13B | 0.9731 | 0.9573 | 0.9647 | - |
| MiningCoT7B | 0.9713 | 0.9647 | 0.9678 | 0.9980 |
| LC-LLM_CoT13B | 0.9471 | 0.9073 | 0.9237 | - |
| LC-LLMNoCoT13B | 0.9550 | 0.9280 | 0.9393 | - |
| MiningNoCoT7B | 0.9677 | 0.9507 | 0.9586 | - |
| LC-LLMNoCoT7B | 0.9421 | 0.8947 | 0.9139 | - |
| LC-LLM_CoT7B | 0.9450 | 0.9087 | 0.9236 | - |

</details>

<details open>
<summary>Scenario2</summary>

| Model | Intention Macro-P | Intention Macro-R | Intention Macro-F1 | Load-State Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| MiningCoT13B | 0.7325 | 0.7907 | 0.7130 | 0.9730 |
| LC-LLMNoCoT7B | 0.7942 | 0.8453 | 0.8032 | - |
| MiningNoCoT13B | 0.4912 | 0.6533 | 0.6543 | - |
| MiningCoT7B | 0.8070 | 0.8567 | 0.8149 | 0.9700 |
| MiningNoCoT7B | 0.7199 | 0.7693 | 0.6812 | - |
| LC-LLM_CoT7B | 0.7309 | 0.7820 | 0.6931 | - |
| LC-LLM_CoT13B | 0.7102 | 0.4260 | 0.3096 | - |
| LC-LLMNoCoT13B | 0.7853 | 0.8318 | 0.7884 | - |

</details>

<details open>
<summary>Scenario3</summary>

| Model | Intention Macro-P | Intention Macro-R | Intention Macro-F1 | Load-State Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| MiningCoT13B | 0.6816 | 0.7260 | 0.6717 | 0.9700 |
| LC-LLM_CoT13B | 0.6726 | 0.4220 | 0.2983 | - |
| MiningNoCoT7B | 0.6817 | 0.7413 | 0.6579 | - |
| LC-LLMNoCoT13B | 0.7849 | 0.8087 | 0.7861 | - |
| MiningNoCoT13B | 0.6653 | 0.6593 | 0.4467 | - |
| MiningCoT7B | 0.8116 | 0.8367 | 0.8204 | 0.9699 |
| LC-LLM_CoT7B | 0.7045 | 0.7587 | 0.6931 | - |
| LC-LLMNoCoT7B | 0.7748 | 0.7980 | 0.7843 | - |

</details>

<details open>
<summary>Scenario4</summary>

| Model | Intention Macro-P | Intention Macro-R | Intention Macro-F1 | Load-State Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| MiningCoT13B | 0.6562 | 0.7033 | 0.6303 | 0.9720 |
| LC-LLMNoCoT7B | 0.7035 | 0.7504 | 0.7106 | - |
| MiningNoCoT13B | 0.4614 | 0.6147 | 0.6149 | - |
| LC-LLM_CoT7B | 0.6529 | 0.7047 | 0.6221 | - |
| MiningNoCoT7B | 0.6655 | 0.7073 | 0.6252 | - |
| MiningCoT7B | 0.7219 | 0.7613 | 0.7246 | 0.9720 |
| LC-LLMNoCoT13B | 0.7209 | 0.7653 | 0.7218 | - |
| LC-LLM_CoT13B | 0.5755 | 0.4587 | 0.4829 | - |

</details>

Full machine-readable result files are available in:

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
  author = {Siyu Teng, Jing Wang, Ran Yan, Junyuan Lin, Xiaotong Zhang, Yuchen Li, Mingxing Peng, Meixin Zhu, Jiasong Zhu, Long Chen},
  journal = {TBD},
  year   = {2026}
}
```

## License

The license for this repository will be clarified in a future update. Unless a license file is added, please contact the authors before using the dataset or code for commercial purposes.
