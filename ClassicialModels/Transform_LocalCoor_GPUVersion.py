import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# 1. 加载并处理 JSON 数据
def load_json_as_tensor_pairs(json_path):
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
        raise ValueError("❌ 有效帧不足 1000，请检查数据格式和长度。")

    inputs = torch.tensor(inputs, dtype=torch.float32)  # [N, 12, 2]
    outputs = torch.tensor(outputs, dtype=torch.float32)  # [N, 6, 2]

    # 全局归一化
    all_coords = torch.cat([inputs.reshape(-1, 2), outputs.reshape(-1, 2)], dim=0).numpy()
    scaler = MinMaxScaler()
    scaler.fit(all_coords)

    inputs = torch.tensor(scaler.transform(inputs.reshape(-1, 2)).reshape(inputs.shape), dtype=torch.float32)
    outputs = torch.tensor(scaler.transform(outputs.reshape(-1, 2)).reshape(outputs.shape), dtype=torch.float32)

    return inputs, outputs, scaler


# 新增：仅解析未缩放的数据集（用于独立的训练/测试加载）
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

    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = torch.tensor(outputs, dtype=torch.float32)
    return inputs, outputs


# 新增：对 (X, y) 进行同步缩放；若未提供 scaler，则基于 (X, y) 拟合
def scale_pairs(X, Y, scaler: MinMaxScaler = None):
    if scaler is None:
        all_coords = torch.cat([X.reshape(-1, 2), Y.reshape(-1, 2)], dim=0).numpy()
        scaler = MinMaxScaler().fit(all_coords)
    Xs = torch.tensor(scaler.transform(X.reshape(-1, 2)).reshape(X.shape), dtype=torch.float32)
    Ys = torch.tensor(scaler.transform(Y.reshape(-1, 2)).reshape(Y.shape), dtype=torch.float32)
    return Xs, Ys, scaler


# 2. 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        pe = self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        return x + pe


# 3. Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size=2, d_model=64, nhead=4, num_layers=2, output_size=2, future_steps=6):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(d_model, output_size * future_steps)
        self.future_steps = future_steps
        self.output_size = output_size

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x.view(-1, self.future_steps, self.output_size)


# 4. 训练函数（支持 mini-batch）
def train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, batch_size=512, device=None):
    dataset = TensorDataset(X_train, y_train)
    # 启用 pin_memory 提升 GPU 传输效率（仅在 CUDA 可用时有用）
    pin_mem = bool(device and torch.device(device).type == "cuda")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            if device is not None:
                xb = xb.to(device, non_blocking=pin_mem)
                yb = yb.to(device, non_blocking=pin_mem)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


# 5. 测试与保存预测
def test_model(model, X_test, y_test, scaler, save_path="Output/SG_Transformer_predictions_output.json", device=None):
    model.eval()
    predictions = []

    with torch.no_grad():
        if device is not None:
            X_test = X_test.to(device)
        output = model(X_test)

        # 在转换为 numpy 前先搬回 CPU
        output_np = output.detach().cpu().numpy().reshape(-1, 2)
        y_test_np = y_test.detach().cpu().numpy().reshape(-1, 2)

        output_np = scaler.inverse_transform(output_np)
        y_test_np = scaler.inverse_transform(y_test_np)

        output_np = output_np.reshape(-1, 6, 2)
        y_test_np = y_test_np.reshape(-1, 6, 2)

        for pred, label in zip(output_np, y_test_np):
            predictions.append({"prediction": pred.tolist(), "label": label.tolist()})

        with open(save_path, "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"\n✅ Predictions saved to: {save_path}")

        # RMSE
        rmse_lon = np.sqrt(mean_squared_error(y_test_np[:, :, 0].flatten(), output_np[:, :, 0].flatten()))
        rmse_lat = np.sqrt(mean_squared_error(y_test_np[:, :, 1].flatten(), output_np[:, :, 1].flatten()))
        print(f"📈 RMSE for Longitude: {rmse_lon:.6f}")
        print(f"📈 RMSE for Latitude:  {rmse_lat:.6f}")


# 新增：独立入口，按“加载训练集→训练→保存模型→加载测试集→测试”流程执行
def main_separate_datasets(train_file_path,
                           test_file_path,
                           model_save_path="Output/Transformer_Local_Model.pth",
                           pred_save_path="Output/Transformer_predictions.json"):
    np.random.seed(42)
    torch.manual_seed(42)

    # 设备
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        print("Using GPU: cuda:0")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # 1) 训练集：解析未缩放数据
    X_train_raw, y_train_raw = parse_json_as_tensor_pairs(train_file_path)
    print(f"✅ 训练集帧数: {len(X_train_raw)}")

    # 2) 用训练集拟合 scaler 并缩放
    X_train, y_train, scaler = scale_pairs(X_train_raw, y_train_raw, scaler=None)

    # 3) 构建并训练模型
    model = TransformerModel(input_size=2, d_model=64, nhead=4, num_layers=2, output_size=2, future_steps=6)
    model = model.to(device)
    train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, batch_size=512, device=device)

    # 4) 保存模型
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "input_size": 2,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "output_size": 2,
                "future_steps": 6,
            }
        }, model_save_path)
        print(f"✅ Model saved to: {model_save_path}")

    # 5) 测试集：解析未缩放数据并使用训练集的 scaler 缩放
    X_test_raw, y_test_raw = parse_json_as_tensor_pairs(test_file_path)
    print(f"✅ 测试集帧数: {len(X_test_raw)}")
    X_test, y_test, _ = scale_pairs(X_test_raw, y_test_raw, scaler=scaler)

    # 6) 测试与保存预测
    if pred_save_path:
        os.makedirs(os.path.dirname(pred_save_path) or ".", exist_ok=True)
    test_model(model, X_test, y_test, scaler, save_path=pred_save_path, device=device)


# 6. 主函数
def main(file_path):
    np.random.seed(42)
    torch.manual_seed(42)
    # 单 GPU 设备选择（仅使用 cuda:0）
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        print("Using GPU: cuda:0")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    X, y, scaler = load_json_as_tensor_pairs(file_path)
    total = len(X)

    test_indices = np.random.choice(total, size=1000, replace=False)
    mask = torch.ones(total, dtype=torch.bool)
    mask[test_indices] = False

    X_test = X[test_indices]
    y_test = y[test_indices]
    X_train = X[mask]
    y_train = y[mask]

    print(f"✅ 训练集帧数: {len(X_train)}")
    print(f"✅ 测试集帧数: {len(X_test)}")

    model = TransformerModel(input_size=2, d_model=64, nhead=4, num_layers=2, output_size=2, future_steps=6)
    model = model.to(device)

    train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, batch_size=512, device=device)
    # test_model(
    #     model,
    #     X_test,
    #     y_test,
    #     scaler,
    #     save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/Transformer_SG_predictions_output.json",
    # )
    test_model(
        model,
        X_test,
        y_test,
        scaler,
        device=device,
        # save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/Transformer_Zhoushan_predictions_output.json",
        # save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/Transformer_predictions_0_5draught.json",
        save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/Transformer_predictions_5_10draught.json",
        # save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/Transformer_predictions_10_25draught.json",
    )
    # test_model(
    #     model,
    #     X_test,
    #     y_test,
    #     scaler,
    #     save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/Transformer_EMS_training_6outputs.json",
    # )


# 7. 启动入口
if __name__ == "__main__":
    # 请按需修改以下数据与输出路径
    train_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/draught_5_10m_6outputs.json"
    test_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/draught_10_25m_6outputs.json"

    model_path = "Output/Transformer_Local_Model.pth"
    pred_path = "Output/Transformer_predictions.json"

    main_separate_datasets(train_file, test_file, model_path, pred_path)
