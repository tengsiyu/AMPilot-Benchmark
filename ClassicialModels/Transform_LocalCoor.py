import json
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
def train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, batch_size=512):
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
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
def test_model(model, X_test, y_test, scaler, save_path="Output/SG_Transformer_predictions_output.json"):
    model.eval()
    predictions = []

    with torch.no_grad():
        output = model(X_test)

        output_np = output.numpy().reshape(-1, 2)
        y_test_np = y_test.numpy().reshape(-1, 2)

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


# 6. 主函数
def main(file_path):
    np.random.seed(42)
    torch.manual_seed(42)

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

    train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, batch_size=512)
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
        # save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/Transformer_Zhoushan_predictions_output.json",
        # save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/Transformer_predictions_0_5draught.json",
        save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/Transformer_predictions_5_10draught.json",
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
    # file_path = "/file_system/vepfs/algorithm/siyu.teng/data/SG_training_6outputs.json" # Done
    # file_path = "/file_system/vepfs/algorithm/siyu.teng/data/Zhoushan_training_6outputs.json"
    # file_path = "/file_system/vepfs/algorithm/siyu.teng/data/EMS_training_6outputs.json" # Done
    file_path = "/file_system/vepfs/algorithm/siyu.teng/MarineData/draught_5_10m_6outputs.json"  # Done
    main(file_path)
