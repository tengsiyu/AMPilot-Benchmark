import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import importlib.util


def load_class_from_py(file_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, f"Cannot load {file_path}"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return getattr(module, class_name)


def parse_json_as_tensor_pairs(json_path):
    inputs, outputs = [], []
    with open(json_path, "r") as f:
        data = json.load(f)
    for frame in data:
        try:
            hist = frame["input"]["Ego-ship"]["Historical Trajectory"]
            futu = frame["output"]["Trajectory"]
            if len(hist) < 2 or len(futu) != 6:
                continue
            hist = hist[1:] + [[0.0, 0.0]]
            if len(hist) != 12:
                continue
            inputs.append(hist)
            outputs.append(futu)
        except KeyError:
            continue
    if len(inputs) < 1000:
        raise ValueError("有效帧不足 1000，请检查数据格式和长度")
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)


def scale_pairs(X, Y, scaler: MinMaxScaler = None):
    if scaler is None:
        all_coords = torch.cat([X.reshape(-1, 2), Y.reshape(-1, 2)], dim=0).numpy()
        scaler = MinMaxScaler().fit(all_coords)
    Xs = torch.tensor(scaler.transform(X.reshape(-1, 2)).reshape(X.shape), dtype=torch.float32)
    Ys = torch.tensor(scaler.transform(Y.reshape(-1, 2)).reshape(Y.shape), dtype=torch.float32)
    return Xs, Ys, scaler


def train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, batch_size=512, device=None):
    dataset = TensorDataset(X_train, y_train)
    pin_mem = bool(device and torch.device(device).type == "cuda")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device) if device is not None else xb
            yb = yb.to(device) if device is not None else yb
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(loader):.4f}")


def test_model(model, X_test, y_test, scaler, save_path, device=None):
    model.eval()
    with torch.no_grad():
        Xb = X_test.to(device) if device is not None else X_test
        out = model(Xb)
        out_np = out.detach().cpu().numpy().reshape(-1, 2)
        y_np = y_test.detach().cpu().numpy().reshape(-1, 2)
        out_np = scaler.inverse_transform(out_np)
        y_np = scaler.inverse_transform(y_np)
        out_np = out_np.reshape(-1, 6, 2)
        y_np = y_np.reshape(-1, 6, 2)
        preds = [{"prediction": p.tolist(), "label": t.tolist()} for p, t in zip(out_np, y_np)]
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(preds, f, indent=2)
        print(f"\n✅ Predictions saved to: {save_path}")


def main_separate_datasets(train_file_path,
                           test_file_path,
                           model_save_path="Output/BiIFNet_Model.pth",
                           pred_save_path="Output/BiIFNet_predictions.json"):
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        print("Using GPU: cuda:0")
    else:
        print("CUDA not available, using CPU")

    X_train_raw, y_train_raw = parse_json_as_tensor_pairs(train_file_path)
    print(f"✅ 训练集帧数: {len(X_train_raw)}")
    X_train, y_train, scaler = scale_pairs(X_train_raw, y_train_raw, scaler=None)

    base = os.path.dirname(__file__)
    cls = load_class_from_py(os.path.join(base, "Bi_IFNet_local.py"), "BiIFNet")
    model = cls(input_dim=2, hidden_dim=64, future_steps=6).to(device)
    train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, batch_size=512, device=device)

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"input_dim": 2, "hidden_dim": 64, "future_steps": 6}},
               model_save_path)
    print(f"✅ Model saved to: {model_save_path}")

    X_test_raw, y_test_raw = parse_json_as_tensor_pairs(test_file_path)
    print(f"✅ 测试集帧数: {len(X_test_raw)}")
    X_test, y_test, _ = scale_pairs(X_test_raw, y_test_raw, scaler=scaler)
    test_model(model, X_test, y_test, scaler, save_path=pred_save_path, device=device)


if __name__ == "__main__":
    train_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/draught_5_10m_6outputs.json"
    test_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/draught_10_25m_6outputs.json"
    model_path = "Output/BiIFNet_Model.pth"
    pred_path = "Output/BiIFNet_predictions.json"
    main_separate_datasets(train_file, test_file, model_path, pred_path)

