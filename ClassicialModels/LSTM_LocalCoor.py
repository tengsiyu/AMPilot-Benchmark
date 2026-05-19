import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2, num_layers=1, future_steps=6):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * future_steps)
        self.future_steps = future_steps
        self.output_size = output_size

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.future_steps, self.output_size)


def train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, device=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if device is not None:
            Xb, yb = X_train.to(device), y_train.to(device)
        else:
            Xb, yb = X_train, y_train
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


def test_model(model, X_test, y_test, scaler, save_path="Output/LSTM_predictions.json", device=None):
    model.eval()
    with torch.no_grad():
        if device is not None:
            X_test = X_test.to(device)
        out = model(X_test)
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
                           model_save_path="Output/LSTM_Model.pth",
                           pred_save_path="Output/LSTM_predictions.json"):
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
    model = LSTMModel()
    model = model.to(device)
    train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, device=device)

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"input_size": 2, "hidden_size": 64, "output_size": 2, "num_layers": 1, "future_steps": 6}},
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
    model_path = "Output/LSTM_Model.pth"
    pred_path = "Output/LSTM_predictions.json"
    main_separate_datasets(train_file, test_file, model_path, pred_path)

