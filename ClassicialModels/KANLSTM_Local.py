import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


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


class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(KANLayer, self).__init__()
        self.functions = nn.ModuleList(
            [nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1)) for _ in range(input_dim * output_dim)]
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        B, T, D = x.size()
        x_flat = x.view(-1, D)
        outputs = []
        for i in range(self.output_dim):
            out_i = 0
            for j in range(self.input_dim):
                f = self.functions[i * self.input_dim + j]
                out_i += f(x_flat[:, j:j+1])
            outputs.append(out_i)
        x_out = torch.cat(outputs, dim=1)
        return x_out.view(B, T, self.output_dim)


class KANLSTMModel(nn.Module):
    def __init__(self, input_size=2, kan_hidden=32, lstm_hidden=64, future_steps=6):
        super(KANLSTMModel, self).__init__()
        self.kan = KANLayer(input_size, kan_hidden)
        self.lstm = nn.LSTM(input_size=kan_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, future_steps * 2)
        self.future_steps = future_steps

    def forward(self, x):
        x = self.kan(x)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        out = self.fc(h_last)
        return out.view(-1, self.future_steps, 2)


def train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, batch_size=512, device=None):
    dataset = TensorDataset(X_train, y_train)
    pin_mem = bool(device and torch.device(device).type == "cuda")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device) if device is not None else xb
            yb = yb.to(device) if device is not None else yb
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(loader):.6f}")


def test_model(model, X_test, y_test, scaler, save_path="Output/KANLSTM_predictions.json", device=None):
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
        rmse_lon = np.sqrt(mean_squared_error(y_np[:, :, 0].flatten(), out_np[:, :, 0].flatten()))
        rmse_lat = np.sqrt(mean_squared_error(y_np[:, :, 1].flatten(), out_np[:, :, 1].flatten()))
        print(f"📈 RMSE for Longitude: {rmse_lon:.6f}")
        print(f"📈 RMSE for Latitude:  {rmse_lat:.6f}")


def main_separate_datasets(train_file_path,
                           test_file_path,
                           model_save_path="Output/KANLSTM_Model.pth",
                           pred_save_path="Output/KANLSTM_predictions.json"):
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
    model = KANLSTMModel()
    model = model.to(device)
    train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, batch_size=512, device=device)

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"input_size": 2, "kan_hidden": 32, "lstm_hidden": 64, "future_steps": 6}},
               model_save_path)
    print(f"✅ Model saved to: {model_save_path}")

    X_test_raw, y_test_raw = parse_json_as_tensor_pairs(test_file_path)
    print(f"✅ 测试集帧数: {len(X_test_raw)}")
    X_test, y_test, _ = scale_pairs(X_test_raw, y_test_raw, scaler=scaler)
    if pred_save_path:
        os.makedirs(os.path.dirname(pred_save_path) or ".", exist_ok=True)
    test_model(model, X_test, y_test, scaler, save_path=pred_save_path, device=device)


if __name__ == "__main__":
    train_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/draught_5_10m_6outputs.json"
    test_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/draught_10_25m_6outputs.json"
    model_path = "Output/KANLSTM_Model.pth"
    pred_path = "Output/KANLSTM_predictions.json"
    main_separate_datasets(train_file, test_file, model_path, pred_path)
