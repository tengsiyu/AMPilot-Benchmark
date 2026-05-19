import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MultiModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, h = self.rnn(x)
        return out, h[-1]


class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_modalities):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.keys = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_modalities)])
        self.values = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_modalities)])

    def forward(self, h_main, hs):
        attention_scores = []
        for i, h in enumerate(hs):
            score = torch.sum(self.query(h_main) * self.keys[i](h), dim=-1)
            attention_scores.append(score)
        weights = F.softmax(torch.stack(attention_scores, dim=1), dim=1)
        values = torch.stack([self.values[i](h) for i, h in enumerate(hs)], dim=1)
        context = torch.sum(weights.unsqueeze(-1) * values, dim=1)
        return context


class TrajectoryDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, future_steps):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
        self.future_steps = future_steps

    def forward(self, context):
        pred = []
        x = torch.zeros(context.size(0), 1, context.size(1), device=context.device)
        h = context.unsqueeze(0)
        for _ in range(self.future_steps):
            out, h = self.rnn(x, h)
            p = self.fc(out.squeeze(1))
            pred.append(p)
            x = out
        return torch.stack(pred, dim=1)


class MMTPModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, future_steps=6):
        super().__init__()
        self.enc1 = MultiModalEncoder(input_dim, hidden_dim)
        self.enc2 = MultiModalEncoder(input_dim, hidden_dim)
        self.attn = AttentionFusion(hidden_dim, num_modalities=2)
        self.decoder = TrajectoryDecoder(hidden_dim, hidden_dim, future_steps)

    def forward(self, x_main, x_neighbor):
        _, h1 = self.enc1(x_main)
        _, h2 = self.enc2(x_neighbor)
        context = self.attn(h1, [h1, h2])
        return self.decoder(context)


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
            pred = model(xb, xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")


def test_model(model, X_test, y_test, scaler, save_path="Output/MMTP_predictions.json", device=None):
    model.eval()
    with torch.no_grad():
        Xb = X_test.to(device) if device is not None else X_test
        pred = model(Xb, Xb)
        output_np = pred.detach().cpu().numpy().reshape(-1, 2)
        y_test_np = y_test.detach().cpu().numpy().reshape(-1, 2)
        output_np = scaler.inverse_transform(output_np)
        y_test_np = scaler.inverse_transform(y_test_np)
        output_np = output_np.reshape(-1, 6, 2)
        y_test_np = y_test_np.reshape(-1, 6, 2)
        preds = [{"prediction": p.tolist(), "label": t.tolist()} for p, t in zip(output_np, y_test_np)]
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(preds, f, indent=2)
        print(f"\n✅ Predictions saved to: {save_path}")
        rmse_lon = np.sqrt(mean_squared_error(y_test_np[:, :, 0].flatten(), output_np[:, :, 0].flatten()))
        rmse_lat = np.sqrt(mean_squared_error(y_test_np[:, :, 1].flatten(), output_np[:, :, 1].flatten()))
        print(f"📈 RMSE for Longitude: {rmse_lon:.6f}")
        print(f"📈 RMSE for Latitude:  {rmse_lat:.6f}")


def main_separate_datasets(train_file_path,
                           test_file_path,
                           model_save_path="Output/MMTP_Model.pth",
                           pred_save_path="Output/MMTP_predictions.json"):
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
    model = MMTPModel()
    model = model.to(device)
    train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, batch_size=512, device=device)

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"input_dim": 2, "hidden_dim": 64, "future_steps": 6}},
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
    model_path = "Output/MMTP_Model.pth"
    pred_path = "Output/MMTP_predictions.json"
    main_separate_datasets(train_file, test_file, model_path, pred_path)

