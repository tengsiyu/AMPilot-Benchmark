import os
import sys
import json
import argparse
import importlib
import importlib.util
import pickle
import numpy as np
import torch
import torch.nn as nn


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# 运行代码
# python /file_system/vepfs/algorithm/siyu.teng/Classical_Model/test_all_model.py --test "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/draught_5_10m_selected_revised_trajectory.json" --model-dir "/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/models_5_10" --pred-dir "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/models_5_10"

# python /file_system/vepfs/algorithm/siyu.teng/Classical_Model/test_all_model.py --test "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/SG_testing_dataset_6outputs_with_instruction.json" --model-dir "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/SG_Areas/Models" --pred-dir "/file_system/vepfs/algorithm/siyu.teng/MarineData/different_water_areas/SG_Areas/Output"

# 　python /file_system/vepfs/algorithm/siyu.teng/Classical_Model/test_all_model.py --test "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/diff_area_EMS_selected_revised_trajectory.json" --model-dir "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/models_diffarea_EMS/Models" --pred-dir "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/models_diffarea_EMS/Output"

# 　python /file_system/vepfs/algorithm/siyu.teng/Classical_Model/test_all_model.py --test "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/draught_10_25m_selected_revised_trajectory.json" --model-dir "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/models_10_25" --pred-dir "/file_system/vepfs/algorithm/siyu.teng/MarineData/single_traj_test/models_10_25/Output"

# python /file_system/vepfs/algorithm/siyu.teng/Classical_Model/test_all_model.py --test "/file_system/vepfs/algorithm/siyu.teng/AMPolit/Dataset260416/New_scenario/MiningCoT_Scenario1_test_5000.json" --model-dir "/file_system/vepfs/algorithm/siyu.teng/AMPolit/Dataset260416/Classical_model/Models" --pred-dir "/file_system/vepfs/algorithm/siyu.teng/AMPolit/Dataset260416/Classical_model/PredictionScenario1"


def add_path(p: str):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)


BASE_DIR = os.path.dirname(__file__)
DEFAULT_TEST_RELATIVE_PATH = os.path.join(
    "datacollect",
    "Cot_data",
    "PengCoT_fixed_0416.json",
)
DEFAULT_MODEL_RELATIVE_DIR = os.path.join("Classical_Model_0423", "Output", "Models")
DEFAULT_PRED_RELATIVE_DIR = os.path.join("Classical_Model_0423", "Output", "Scenario0Predictions")
DEFAULT_TEST_SIZE = 5000


def find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, "datacollect")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start_dir)
        cur = parent


REPO_ROOT = find_repo_root(BASE_DIR)
DEFAULT_TEST_JSON = (
    r"/file_system/vepfs/algorithm/siyu.teng/AMPolit/Dataset260416/New_scenario/MiningCoT_Scenario3_test_5000.json"
)
DEFAULT_MODEL_DIR = r"/file_system/vepfs/algorithm/siyu.teng/AMPolit/Dataset260416/Classical_model/Models"
DEFAULT_PRED_DIR = r"/file_system/vepfs/algorithm/siyu.teng/AMPolit/Dataset260416/Classical_model/PredictionScenario3"


def resolve_path(path: str, *, must_exist: bool = False) -> str:
    """Resolve absolute paths or repo-relative/script-relative paths."""
    if os.path.isabs(path):
        resolved = path
    else:
        candidates = [
            os.path.join(os.getcwd(), path),
            os.path.join(REPO_ROOT, path),
            os.path.join(BASE_DIR, path),
        ]
        resolved = next((p for p in candidates if os.path.exists(p)), candidates[0])

    resolved = os.path.abspath(resolved)
    if must_exist and not os.path.exists(resolved):
        raise SystemExit(f"[Error] Path not found: {resolved}")
    return resolved


add_path(BASE_DIR)

parse_json_as_tensor_pairs = None
scale_pairs = None
build_model = None
maybe_compile = None
maybe_data_parallel = None
train_one = None
test_one = None
training_utils_module = None


def load_training_utils(train_utils_dir: str | None = None):
    """Load shared model/data helpers from train_all_models_AMPolit.py when needed."""
    global parse_json_as_tensor_pairs
    global scale_pairs
    global build_model
    global maybe_compile
    global maybe_data_parallel
    global train_one
    global test_one
    global training_utils_module

    train_utils_base = resolve_path(train_utils_dir, must_exist=True) if train_utils_dir else BASE_DIR
    train_utils_path = os.path.join(train_utils_base, "train_all_models_AMPolit.py")
    if not os.path.isfile(train_utils_path):
        raise SystemExit(
            f"[Error] Cannot find train_all_models_AMPolit.py at {train_utils_path}. "
            "Pass --train-utils-dir <directory-containing-train_all_models_AMPolit.py>."
        )

    spec = importlib.util.spec_from_file_location("ampolit_train_utils", train_utils_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[Error] Cannot load train utils from: {train_utils_path}")
    train_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_utils)
    print(f"[Info] Loaded training utilities: {train_utils_path}")

    parse_json_as_tensor_pairs = train_utils.parse_json_as_tensor_pairs
    scale_pairs = train_utils.scale_pairs
    build_model = train_utils.build_model
    maybe_compile = train_utils.maybe_compile
    maybe_data_parallel = train_utils.maybe_data_parallel
    train_one = train_utils.train_one
    test_one = train_utils.test_one
    training_utils_module = train_utils


def _load_scaler_from_dir(model_dir: str):
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.isfile(scaler_path):
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None


def normalize_state_dict_keys(state_dict):
    prefixes = ("_orig_mod.", "module.")
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        normalized[new_key] = value
    return normalized


def infer_future_steps_from_state(state_dict):
    for key in ("fc.bias", "fc.weight"):
        tensor = state_dict.get(key)
        if tensor is None:
            continue
        out_features = tensor.shape[0]
        if out_features % 2 == 0:
            return out_features // 2
    return None


def infer_future_steps_from_payload(payload, state_dict, label_steps: int):
    config = payload.get("config") or {}
    future_steps = config.get("future_steps")
    if future_steps is not None:
        return int(future_steps)
    return infer_future_steps_from_state(state_dict) or int(label_steps)


def set_training_future_steps(future_steps: int):
    if training_utils_module is not None and hasattr(training_utils_module, "FUTURE_STEPS"):
        training_utils_module.FUTURE_STEPS = future_steps


def select_test_tensors(source_json: str, explicit_test_json: str | None, test_size: int, seed: int):
    if explicit_test_json:
        return parse_json_as_tensor_pairs(explicit_test_json)

    X_raw, y_raw = parse_json_as_tensor_pairs(source_json)
    if hasattr(training_utils_module, "_train_test_split"):
        _, _, X_test_raw, y_test_raw = training_utils_module._train_test_split(
            X_raw,
            y_raw,
            test_size=test_size,
            seed=seed,
        )
        return X_test_raw, y_test_raw

    rng = np.random.default_rng(seed)
    test_idx = rng.choice(len(X_raw), size=test_size, replace=False)
    return X_raw[test_idx], y_raw[test_idx]


def predict_and_save(model, X_test, y_test, device, scaler, model_kind: str, save_path: str):
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

        pred_steps = int(out.shape[1])
        label_steps = int(y_test.shape[1])

        output_np = out.detach().cpu().numpy().reshape(-1, 2)
        y_test_np = y_test.detach().cpu().numpy().reshape(-1, 2)
        output_np = scaler.inverse_transform(output_np).reshape(-1, pred_steps, 2)
        y_test_np = scaler.inverse_transform(y_test_np).reshape(-1, label_steps, 2)

        compare_steps = min(pred_steps, label_steps)
        if pred_steps != label_steps:
            print(
                f"[Warn] {model_kind}: prediction steps ({pred_steps}) != label steps ({label_steps}); "
                f"metrics use first {compare_steps} steps."
            )

        preds = [
            {
                "prediction": prediction.tolist(),
                "label": label.tolist(),
                "metric_steps": compare_steps,
            }
            for prediction, label in zip(output_np, y_test_np)
        ]
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)
        print(f"[Info] [{model_kind}] Predictions saved to: {save_path}")

        pred_eval = output_np[:, :compare_steps, :]
        label_eval = y_test_np[:, :compare_steps, :]
        rmse_lon = np.sqrt(np.mean((label_eval[:, :, 0] - pred_eval[:, :, 0]) ** 2))
        rmse_lat = np.sqrt(np.mean((label_eval[:, :, 1] - pred_eval[:, :, 1]) ** 2))
        print(f"[Info] [{model_kind}] RMSE Lon: {rmse_lon:.6f}  Lat: {rmse_lat:.6f}")
        return float(rmse_lon), float(rmse_lat), pred_steps, compare_steps


def eval_all_models(
    model_dir: str,
    source_json: str,
    pred_dir: str,
    device: torch.device,
    explicit_test_json: str | None = None,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = 42,
):
    model_dir = resolve_path(model_dir, must_exist=True)
    source_json = resolve_path(source_json, must_exist=True)
    if explicit_test_json:
        explicit_test_json = resolve_path(explicit_test_json, must_exist=True)
    pred_dir = resolve_path(pred_dir, must_exist=False)
    os.makedirs(pred_dir, exist_ok=True)
    print(f"[Info] Source dataset: {source_json}")
    if explicit_test_json:
        print(f"[Info] Test dataset: {explicit_test_json}")
    else:
        print(f"[Info] Test split: seed={seed}, test_size={test_size}")
    print(f"[Info] Model directory: {model_dir}")
    print(f"[Info] Prediction directory: {pred_dir}")
    # Prepare test tensors and scaler
    X_test_raw, y_test_raw = select_test_tensors(source_json, explicit_test_json, test_size, seed)
    if X_test_raw.numel() == 0 or y_test_raw.numel() == 0:
        dataset_label = explicit_test_json or source_json
        raise SystemExit(f"[Error] No valid samples parsed from test dataset: {dataset_label}")
    print(f"[Info] Parsed test tensors: X={tuple(X_test_raw.shape)}, y={tuple(y_test_raw.shape)}")

    scaler = _load_scaler_from_dir(model_dir)
    if scaler is None:
        print(f"[Warn] scaler.pkl not found in {model_dir}. Fitting scaler on test set (not ideal).")
        X_test, y_test, scaler = scale_pairs(X_test_raw, y_test_raw, scaler=None)
    else:
        X_test, y_test, _ = scale_pairs(X_test_raw, y_test_raw, scaler=scaler)

    summary = {}
    evaluated = 0
    # Enumerate .pth files in alphabetical order
    for fn in sorted(os.listdir(model_dir)):
        if not fn.lower().endswith(".pth"):
            continue
        model_path = os.path.join(model_dir, fn)
        try:
            payload = torch.load(model_path, map_location="cpu")
        except Exception as e:
            print(f"[Warn] Skip unreadable model: {model_path} ({e})")
            continue
        kind = payload.get("kind")
        state = payload.get("state_dict")
        if not kind or state is None:
            print(f"[Warn] Missing kind/state in: {model_path}. Skipping.")
            continue

        state = normalize_state_dict_keys(state)
        future_steps = infer_future_steps_from_payload(payload, state, int(y_test_raw.shape[1]))
        set_training_future_steps(future_steps)

        print(f"\n==== Evaluating {kind} from {fn} (future_steps={future_steps}) ====")
        try:
            model = build_model(kind, future_steps=future_steps)
        except TypeError:
            model = build_model(kind)
        model.load_state_dict(state, strict=True)
        model = model.to(device)
        pred_path = os.path.join(pred_dir, f"{os.path.splitext(fn)[0]}_predictions.json")
        rmse_lon, rmse_lat, pred_steps, metric_steps = predict_and_save(
            model,
            X_test,
            y_test,
            device,
            scaler,
            model_kind=kind,
            save_path=pred_path,
        )
        summary[kind] = {
            "rmse_lon": rmse_lon,
            "rmse_lat": rmse_lat,
            "prediction_steps": pred_steps,
            "label_steps": int(y_test_raw.shape[1]),
            "metric_steps": metric_steps,
        }
        evaluated += 1

    if evaluated == 0:
        raise SystemExit(f"[Error] No .pth model files were evaluated under: {model_dir}")

    # Save summary
    with open(os.path.join(pred_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Info] Summary saved to: {os.path.join(pred_dir, 'summary_metrics.json')}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all well-trained models under --model-dir on the same test set; "
            "optionally train+evaluate a single model if --train is provided."
        )
    )
    # Train+eval flags
    parser.add_argument("--train", help="Path to training dataset JSON")
    parser.add_argument("--kind", default="GRU", help="Model kind for training")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=("Directory to save or load model + scaler. Defaults to " "Classical_Model_0423/Output/Models"),
    )
    parser.add_argument(
        "--pred-dir",
        default=DEFAULT_PRED_DIR,
        help=("Directory to save predictions. Defaults to " "Classical_Model_0423/Output/Scenario0Predictions"),
    )
    parser.add_argument(
        "--test",
        default=None,
        help=(
            "Optional explicit test dataset JSON. If omitted, reproduces the train script's "
            "Scenario0 split from --source."
        ),
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_TEST_JSON,
        help="Source dataset for Scenario0 split. Defaults to datacollect/Cot_data/PengCoT_fixed_0416.json.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--device", default="cuda:0", help="Evaluation device. Defaults to cuda:0.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback if CUDA is unavailable.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile if available")
    parser.add_argument("--no-data-parallel", action="store_true", help="Disable DataParallel on multi-GPU")
    parser.add_argument(
        "--train-utils-dir",
        default=None,
        help="Directory containing train_all_models_AMPolit.py if it is not beside this test.py",
    )
    # Eval-all flag
    parser.add_argument(
        "--eval-model-dir",
        default=None,
        help="[Deprecated] If set, evaluate all .pth models under this directory; otherwise uses --model-dir",
    )
    args = parser.parse_args()
    args.source = resolve_path(args.source, must_exist=True)
    if args.test:
        args.test = resolve_path(args.test, must_exist=True)
    if args.train:
        args.train = resolve_path(args.train, must_exist=True)
    args.model_dir = resolve_path(args.model_dir, must_exist=False)
    args.pred_dir = resolve_path(args.pred_dir, must_exist=False)
    if args.eval_model_dir:
        args.eval_model_dir = resolve_path(args.eval_model_dir, must_exist=True)
    if args.train_utils_dir:
        args.train_utils_dir = resolve_path(args.train_utils_dir, must_exist=True)

    load_training_utils(args.train_utils_dir)

    np.random.seed(42)
    torch.manual_seed(42)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise SystemExit(
                "[Error] CUDA device was requested but this Python environment cannot use CUDA. "
                "Install a CUDA-enabled PyTorch build or run with --allow-cpu to bypass."
            )
        args.device = "cpu"

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print("Using CPU")

    # Evaluation-only: when --train is not provided, evaluate all models under model dir
    if not args.train:
        eval_dir = args.eval_model_dir or args.model_dir
        if not os.path.isdir(eval_dir):
            raise SystemExit(f"[Error] Model directory not found: {eval_dir}")
        eval_all_models(
            eval_dir,
            args.source,
            args.pred_dir,
            device,
            explicit_test_json=args.test,
            test_size=args.test_size,
            seed=args.seed,
        )
        return

    # If both training and eval-all are requested, run eval-all first
    if args.eval_model_dir:
        eval_all_models(
            args.eval_model_dir,
            args.source,
            args.pred_dir,
            device,
            explicit_test_json=args.test,
            test_size=args.test_size,
            seed=args.seed,
        )

    # If training is requested, train a single model and evaluate it
    if args.train:
        # Load datasets
        X_train_raw, y_train_raw = parse_json_as_tensor_pairs(args.train)
        X_test_raw, y_test_raw = select_test_tensors(args.source, args.test, args.test_size, args.seed)

        # Fit scaler on train only
        X_train, y_train, scaler = scale_pairs(X_train_raw, y_train_raw, scaler=None)
        X_test, y_test, _ = scale_pairs(X_test_raw, y_test_raw, scaler=scaler)

        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        # Save scaler
        scaler_path = os.path.join(args.model_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to: {scaler_path}")

        kind = args.kind
        print(f"\n==== Training {kind} ====")
        future_steps = int(y_train_raw.shape[1])
        try:
            model = build_model(kind, future_steps=future_steps)
        except TypeError:
            model = build_model(kind)
        model = model.to(device)

        use_dp = not args.no_data_parallel
        enable_compile = not args.no_compile

        if use_dp:
            model = maybe_data_parallel(model)
        if enable_compile and not isinstance(model, nn.DataParallel):
            model = maybe_compile(model)

        train_one(
            model,
            X_train,
            y_train,
            device,
            model_kind=kind,
            epochs=int(args.epochs),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
        )

        # Save model weights
        model_path = os.path.join(args.model_dir, f"{kind}.pth")
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({"state_dict": state, "kind": kind}, model_path)
        print(f"Model saved to: {model_path}")

        # Evaluate and save predictions
        pred_path = os.path.join(args.pred_dir, f"{kind}_predictions.json")
        rmse_lon, rmse_lat = test_one(model, X_test, y_test, device, scaler, model_kind=kind, save_path=pred_path)
        print(json.dumps({"rmse_lon": rmse_lon, "rmse_lat": rmse_lat}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
