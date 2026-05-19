import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
import importlib.util
import ast
import inspect
from typing import Optional


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(__file__)


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, "AMPolit")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start_dir)
        cur = parent


REPO_ROOT = _find_repo_root(BASE_DIR)
print(f"Repository root detected at: {REPO_ROOT}")
DEFAULT_TRAIN_FILE = r"/file_system/vepfs/algorithm/siyu.teng/AMPolit/Dataset260416/MiningCoT_fixed_0416.json"
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "Classical_Model", "Output", "Models")
DEFAULT_PRED_DIR = os.path.join(REPO_ROOT, "Classical_Model", "Output", "Scenario0Predictions")
DEFAULT_TEST_SIZE = 5000


def _read_json_frames(json_path):
    """Robustly read frames from JSON-like files.
    Supports: JSON list, dict-wrapped list, list of JSON-strings, or NDJSON (one JSON per line).
    Returns a list of frame dicts.
    """
    # Try standard json.load first
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # Fallback to NDJSON
        frames = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    frames.append(json.loads(s))
                except Exception:
                    continue
        return frames

    # If dict-wrapped, try common container keys or first list value
    if isinstance(data, dict):
        for key in ("data", "frames", "items", "records", "list"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # fallback: first list value
        for v in data.values():
            if isinstance(v, list):
                return v
        return []

    # If list, but elements might be strings
    if isinstance(data, list):
        if data and isinstance(data[0], str):
            frames = []
            for s in data:
                try:
                    frames.append(json.loads(s))
                except Exception:
                    continue
            return frames
        return data

    # Unknown structure
    return []


def _extract_list_after_label(text: str, label: str):
    if not isinstance(text, str):
        return None
    idx = text.rfind(label)
    if idx < 0:
        return None
    sub = text[idx + len(label) :]
    start = sub.find("[")
    if start < 0:
        return None
    start = idx + len(label) + start
    depth = 0
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    try:
        return ast.literal_eval(text[start:end])
    except Exception:
        return None


def _extract_list_after_labels(text: str, labels):
    for label in labels:
        result = _extract_list_after_label(text, label)
        if result is not None:
            return result
    return None


def parse_json_as_tensor_pairs(json_path):
    inputs, outputs = [], []
    frames = _read_json_frames(json_path)
    skipped = 0
    target_hist_len = None
    target_futu_len = None
    for frame in frames:
        if isinstance(frame, str):
            try:
                frame = json.loads(frame)
            except Exception:
                skipped += 1
                continue
        try:
            hist = None
            futu = None
            if isinstance(frame.get("input"), dict):
                hist = frame["input"]["Ego-ship"]["Historical Trajectory"]
            elif isinstance(frame.get("input"), str):
                hist = _extract_list_after_labels(
                    frame["input"],
                    [
                        "Historical Trajectory",
                        "Historical position of the last 2 seconds",
                        "Historical position",
                    ],
                )
            if isinstance(frame.get("output"), dict):
                futu = frame["output"]["Trajectory"]
            elif isinstance(frame.get("output"), str):
                futu = _extract_list_after_labels(
                    frame["output"],
                    [
                        "Trajectory",
                        "Final Answer:\n - Trajectory",
                        "Final Answer:\n - Trajectory:",
                    ],
                )
            if hist is None or futu is None:
                skipped += 1
                continue
            if len(hist) < 2 or len(futu) < 1:
                skipped += 1
                continue
            if target_hist_len is None:
                target_hist_len = len(hist)
            if target_futu_len is None:
                target_futu_len = len(futu)
            if len(hist) != target_hist_len or len(futu) != target_futu_len:
                skipped += 1
                continue
            inputs.append(hist)
            outputs.append(futu)
        except Exception:
            skipped += 1
            continue
    if len(inputs) == 0:
        raise ValueError(
            f"No valid samples parsed from {json_path}. "
            f"total_frames={len(frames)}, skipped={skipped}. "
            "Check the JSON structure and required keys: "
            "input->Ego-ship->Historical Trajectory and output->Trajectory."
        )
    X = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(outputs, dtype=torch.float32)
    return X, y


def scale_pairs(X, Y, scaler: MinMaxScaler = None):
    if scaler is None:
        all_coords = torch.cat([X.reshape(-1, 2), Y.reshape(-1, 2)], dim=0).numpy()
        scaler = MinMaxScaler().fit(all_coords)
    Xs = torch.tensor(scaler.transform(X.reshape(-1, 2)).reshape(X.shape), dtype=torch.float32)
    Ys = torch.tensor(scaler.transform(Y.reshape(-1, 2)).reshape(Y.shape), dtype=torch.float32)
    return Xs, Ys, scaler


def load_class_from_py(file_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, f"Cannot load {file_path}"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    cls = getattr(module, class_name)
    return cls


def maybe_compile(model: torch.nn.Module):
    try:
        model = torch.compile(model)  # type: ignore[attr-defined]
    except Exception:
        pass
    return model


def maybe_data_parallel(model: torch.nn.Module):
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n and n > 1:
            print(f"Using DataParallel across {n} GPUs")
            model = nn.DataParallel(model, device_ids=list(range(n)))
    return model


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.join(os.getcwd(), path),
        os.path.join(REPO_ROOT, path),
        os.path.join(BASE_DIR, path),
    ]
    return next((candidate for candidate in candidates if os.path.exists(candidate)), candidates[0])


def _train_test_split(X, y, test_size: int, seed: int):
    total = len(X)
    if total == 0:
        raise ValueError("Empty dataset: no samples parsed from input file.")
    if test_size <= 0:
        raise ValueError("test_size must be > 0.")
    if total <= test_size:
        raise ValueError(f"Not enough samples for test split: total={total}, test_size={test_size}.")
    rng = np.random.default_rng(seed)
    test_idx = rng.choice(total, size=test_size, replace=False)
    mask = np.ones(total, dtype=bool)
    mask[test_idx] = False
    X_test, y_test = X[test_idx], y[test_idx]
    X_train, y_train = X[mask], y[mask]
    return X_train, y_train, X_test, y_test


def train_one(model, X_train, y_train, device, model_kind: str, epochs=300, lr=1e-3, batch_size=512):
    dataset = TensorDataset(X_train, y_train)
    pin_mem = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=pin_mem)
            yb = yb.to(device, non_blocking=pin_mem)
            optimizer.zero_grad()
            if model_kind == "ST_GCN":
                x_input = xb.unsqueeze(2)
                adj = torch.eye(1, device=device).unsqueeze(0).repeat(xb.size(0), 1, 1)
                pred = model(x_input, adj)
            elif model_kind == "MMTP":
                pred = model(xb, xb)
            else:
                pred = model(xb)
            if pred.ndim == 3 and yb.ndim == 3 and pred.shape[1] != yb.shape[1]:
                steps = min(pred.shape[1], yb.shape[1])
                pred = pred[:, :steps]
                yb = yb[:, :steps]
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"[{model_kind}] Epoch {epoch + 1}, Loss: {epoch_loss / len(loader):.6f}")


def test_one(model, X_test, y_test, device, scaler: MinMaxScaler, model_kind: str, save_path: str):
    model.eval()
    pin_mem = device.type == "cuda"
    with torch.no_grad():
        xb = X_test.to(device, non_blocking=pin_mem)
        if model_kind == "ST_GCN":
            x_input = xb.unsqueeze(2)
            adj = torch.eye(1, device=device).unsqueeze(0).repeat(xb.size(0), 1, 1)
            out = model(x_input, adj)
        elif model_kind == "MMTP":
            out = model(xb, xb)
        else:
            out = model(xb)
        if out.ndim == 3 and y_test.ndim == 3 and out.shape[1] != y_test.shape[1]:
            steps = min(out.shape[1], y_test.shape[1])
            out = out[:, :steps]
            y_test = y_test[:, :steps]

        output_np = out.detach().cpu().numpy().reshape(-1, 2)
        y_test_np = y_test.detach().cpu().numpy().reshape(-1, 2)
        output_np = scaler.inverse_transform(output_np)
        y_test_np = scaler.inverse_transform(y_test_np)
        steps = out.shape[1] if out.ndim == 3 else y_test.shape[1]
        output_np = output_np.reshape(-1, steps, 2)
        y_test_np = y_test_np.reshape(-1, steps, 2)

        preds = [{"prediction": p.tolist(), "label": t.tolist()} for p, t in zip(output_np, y_test_np)]
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)
        print(f"✅ [{model_kind}] Predictions saved to: {save_path}")

        rmse_lon = np.sqrt(mean_squared_error(y_test_np[:, :, 0].flatten(), output_np[:, :, 0].flatten()))
        rmse_lat = np.sqrt(mean_squared_error(y_test_np[:, :, 1].flatten(), output_np[:, :, 1].flatten()))
        print(f"📈 [{model_kind}] RMSE Lon: {rmse_lon:.6f}  Lat: {rmse_lat:.6f}")
        return float(rmse_lon), float(rmse_lat)


def build_model(kind: str, future_steps: Optional[int] = None):
    base = os.path.dirname(__file__)
    mapping = {
        "Transformer": (os.path.join(base, "Transform_LocalCoor_GPUVersion.py"), "TransformerModel", {}),
        "ST_GCN": (
            os.path.join(base, "ST_GCN_Local.py"),
            "STGCNModel",
            {"node_features": 2, "hidden_dim": 64, "output_steps": 6},
        ),
        "GRU": (
            os.path.join(base, "GRU_LocalModel.py"),
            "GRUModel",
            {"input_size": 2, "hidden_size": 64, "output_size": 2, "num_layers": 1, "future_steps": 6},
        ),
        "LSTM": (
            os.path.join(base, "LSTM_LocalCoor.py"),
            "LSTMModel",
            {"input_size": 2, "hidden_size": 64, "output_size": 2, "num_layers": 1, "future_steps": 6},
        ),
        "RNN": (
            os.path.join(base, "RNN_LocalCoor.py"),
            "RNNModel",
            {"input_size": 2, "hidden_size": 64, "output_size": 2, "num_layers": 1, "future_steps": 6},
        ),
        "BiIFNet": (
            os.path.join(base, "Bi_IFNet_local.py"),
            "BiIFNet",
            {"input_dim": 2, "hidden_dim": 64, "future_steps": 6},
        ),
        "KANLSTM": (
            os.path.join(base, "KANLSTM_Local.py"),
            "KANLSTMModel",
            {"input_size": 2, "kan_hidden": 32, "lstm_hidden": 64, "future_steps": 6},
        ),
        "MMTP": (
            os.path.join(base, "MMTP_prediction_Local.py"),
            "MMTPModel",
            {"input_dim": 2, "hidden_dim": 64, "future_steps": 6},
        ),
    }
    py, cls_name, kwargs = mapping[kind]
    cls = load_class_from_py(py, cls_name)
    if future_steps is not None:
        try:
            params = inspect.signature(cls.__init__).parameters
            for key in ("future_steps", "output_steps", "pred_steps", "n_future"):
                if key in params:
                    kwargs[key] = future_steps
        except Exception:
            pass
    model = cls(**kwargs)
    return model


def main(
    train_file: str,
    test_file: str = None,
    out_dir_models: str = "Output/models_0_5",
    out_dir_preds: str = "Output/preds_0_5",
    epochs: int = 300,
    batch_size: int = 512,
    lr: float = 1e-3,
    enable_compile: bool = True,
    use_data_parallel: bool = True,
    seed: int = 42,
    test_size: int = 1000,
    device_name: str = "cuda:0",
    allow_cpu: bool = False,
):
    _set_seed(seed)

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu:
            raise SystemExit(
                "[Error] CUDA device was requested but this Python environment cannot use CUDA. "
                "Install a CUDA-enabled PyTorch build or run with --allow-cpu to bypass."
            )
        device_name = "cpu"

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print("Using CPU")

    # Load datasets
    train_file = _resolve_path(train_file)
    test_file = _resolve_path(test_file)
    X_train_raw, y_train_raw = parse_json_as_tensor_pairs(train_file)
    if test_file:
        X_test_raw, y_test_raw = parse_json_as_tensor_pairs(test_file)
    else:
        X_train_raw, y_train_raw, X_test_raw, y_test_raw = _train_test_split(
            X_train_raw, y_train_raw, test_size=test_size, seed=seed
        )

    # Fit scaler on train only
    print("X_test_raw shape:", X_test_raw.shape)
    print("y_test_raw shape:", y_test_raw.shape)
    future_steps = int(y_train_raw.shape[1])
    X_train, y_train, scaler = scale_pairs(X_train_raw, y_train_raw, scaler=None)
    X_test, y_test, _ = scale_pairs(X_test_raw, y_test_raw, scaler=scaler)

    os.makedirs(out_dir_models, exist_ok=True)
    os.makedirs(out_dir_preds, exist_ok=True)
    # Save scaler for reproducibility
    with open(os.path.join(out_dir_models, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved to: {os.path.join(out_dir_models, 'scaler.pkl')}")

    kinds = [
        "BiIFNet",
        "KANLSTM",
        "MMTP",
        "Transformer",
        "ST_GCN",
        "GRU",
        "LSTM",
        "RNN",
    ]

    summary = {}
    for kind in kinds:
        print(f"\n==== Training {kind} ====")
        model = build_model(kind, future_steps=future_steps)
        model = model.to(device)
        # 优先使用多卡并行，其次再尝试编译
        if use_data_parallel:
            model = maybe_data_parallel(model)
        if enable_compile and device.type == "cuda" and not isinstance(model, nn.DataParallel):
            model = maybe_compile(model)

        train_one(model, X_train, y_train, device, model_kind=kind, epochs=epochs, lr=lr, batch_size=batch_size)

        # Save model
        model_path = os.path.join(out_dir_models, f"{kind}.pth")
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(
            {
                "state_dict": state,
                "kind": kind,
                "config": {
                    "input_steps": int(X_train_raw.shape[1]),
                    "future_steps": future_steps,
                    "input_size": int(X_train_raw.shape[2]),
                    "output_size": int(y_train_raw.shape[2]),
                    "train_file": train_file,
                    "test_file": test_file,
                },
            },
            model_path,
        )
        print(f"✅ [{kind}] Model saved to: {model_path}")

        # Test + save predictions
        pred_path = os.path.join(out_dir_preds, f"{kind}_predictions.json")
        rmse_lon, rmse_lat = test_one(model, X_test, y_test, device, scaler, model_kind=kind, save_path=pred_path)
        summary[kind] = {"rmse_lon": rmse_lon, "rmse_lat": rmse_lat}

    # Save summary metrics
    with open(os.path.join(out_dir_preds, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ Summary saved to: {os.path.join(out_dir_preds, 'summary_metrics.json')}")


def _legacy_main_disabled():
    # 请按需修改下面路径
    # train_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/draught_0_5m_6outputs.json"
    # test_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/0_5_test_proportional_1000_251027.json"
    # out_dir_models = "/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/models_0_5"
    # out_dir_preds = "/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/preds_0_5"

    # train_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/draught_5_10m_6outputs.json"
    # test_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/5_10_test_proportional_1000_251027_6outputs.json"
    # out_dir_models = "/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/models_5_10"
    # out_dir_preds = "/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/preds_5_10"

    # train_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/draught_10_25m_6outputs.json"
    # test_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/draught_10_25m_selected_revised_trajectory.json"
    # out_dir_models = "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/models_10_25"
    # out_dir_preds = "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/models_10_25/Output"

    # train_file = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/RevisionDataset/PengCoT.json"
    # test_file = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/RevisionDataset/test_dataset_PengCoT_scenario2.json"
    # out_dir_models = (
    #     "/file_system/vepfs/algorithm/siyu.teng/AMPolit/RevisionDataset/PredictResult/ClassicalModels/Peng/Models"
    # )
    # out_dir_preds = (
    #     "/file_system/vepfs/algorithm/siyu.teng/AMPolit/RevisionDataset/PredictResult/ClassicalModels/Peng/Predictions"
    # )

    # train_file = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/RevisionDataset/MiningCoT.json"
    # test_file = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/RevisionDataset/test_dataset_MiningCoT_scenario2.json"
    # out_dir_models = (
    #     "/file_system/vepfs/algorithm/siyu.teng/AMPolit/RevisionDataset/PredictResult/ClassicalModels/Mining/Models"
    # )
    # out_dir_preds = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/RevisionDataset/PredictResult/ClassicalModels/Mining/Predictions"

    # train_file = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/TestData260322/MiningCoT_0322_filtered.json"
    # test_file = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/TestData260322/Test5000_MiningCoT_0322_filtered.json"
    # out_dir_models = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/TestData260322/mining_classical_model/Models"
    # out_dir_preds = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/TestData260322/classical_model/Predictions"

    # train_file = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/TestData260322/PengCoT_0322_filtered.json"
    # test_file = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/TestData260322/Test5000_PengCoT_0322_filtered.json"
    # out_dir_models = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/TestData260322/peng_classical_model/Models"
    # out_dir_preds = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/TestData260322/peng_classical_model/Predictions"

    train_file = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/Dataset260416/PengCoT_fixed_0416.json"
    test_file = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/Dataset260416/PengCoT_fixed_0416_test_5000.json"
    out_dir_models = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/Dataset260416/Classical_model/Models"
    out_dir_preds = "/file_system/vepfs/algorithm/siyu.teng/AMPolit/Dataset260416/Classical_model/Predictions"

    # train_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/SG_training_dataset_6outputs_with_instruction.json"
    # test_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/SG_testing_dataset_6outputs_with_instruction.json"
    # out_dir_models = "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/SG_Areas/Models"
    # out_dir_preds = "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/SG_Areas/Output"

    # train_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/EMS_training_dataset_6outputs_with_instruction.json"
    # test_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/EMS_testing_dataset_6outputs_with_instruction.json"
    # out_dir_models = "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/EMS_Areas/Models"
    # out_dir_preds = "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/EMS_Areas/Output"

    # train_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/Zhoushan_training_dataset_6outputs_with_instruction.json"
    # test_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/EMS_testing_dataset_6outputs_with_instruction.json"
    # out_dir_models = (
    #     "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/models_diffarea_Zhoushan/Models"
    # )
    # out_dir_preds = "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/models_diffarea_Zhoushan/Output"

    main(
        train_file,
        test_file,
        out_dir_models=out_dir_models,
        out_dir_preds=out_dir_preds,
        epochs=300,
        batch_size=1024,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train classical trajectory models on AMPolit data and save predictions. "
            "Defaults use PengCoT_fixed_0416.json and write to Classical_Model/Output."
        )
    )
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE, help="Training dataset JSON.")
    parser.add_argument(
        "--test-file",
        default=None,
        help="Optional test dataset JSON. If omitted, --test-size samples are split from --train-file.",
    )
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR, help="Directory for trained .pth models and scaler.pkl."
    )
    parser.add_argument("--pred-dir", default=DEFAULT_PRED_DIR, help="Directory for prediction JSON and metrics.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--device", default="cuda:0", help="Training device. Defaults to cuda:0.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback if CUDA is unavailable.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--no-data-parallel", action="store_true", help="Disable DataParallel.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.train_file,
        args.test_file,
        out_dir_models=args.model_dir,
        out_dir_preds=args.pred_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        enable_compile=not args.no_compile,
        use_data_parallel=not args.no_data_parallel,
        seed=args.seed,
        test_size=args.test_size,
        device_name=args.device,
        allow_cpu=args.allow_cpu,
    )
