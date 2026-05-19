import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data[["MMSI", "Course", "Speed", "UnixTime", "Lon_d", "Lat_d"]]
    data = data.sort_values(by=["MMSI", "UnixTime"])
    return data


# 数据预处理
def preprocess_data(data, seq_length):
    grouped = data.groupby("MMSI")
    X, y, scalers = [], [], {}

    for mmsi, group in grouped:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(group[["Course", "Speed", "Lon_d", "Lat_d"]])
        scalers[mmsi] = scaler

        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i : i + seq_length])
            y.append(scaled_data[i + seq_length])

    X = np.array(X)
    y = np.array(y)

    # 转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y, scalers


# 位置编码类
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
        batch_size, seq_length, d_model = x.size()  # 获取 batch_size, seq_length 和 d_model
        pe = self.pe[:seq_length, :].unsqueeze(0)  # 形状变为 (1, seq_length, d_model)
        pe = pe.expand(batch_size, -1, -1)  # 扩展到 (batch_size, seq_length, d_model)
        return x + pe  # 直接相加


# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model, dropout=0.1),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()  # 获取 batch_size 和 seq_length
        x = self.input_fc(x)  # 输入到线性层
        x = self.pos_encoder(x)  # 添加位置编码
        x = self.transformer_encoder(x)  # 经过Transformer编码
        x = x.mean(dim=1)  # 取时间序列的平均值作为输出
        x = self.fc(x)  # 最后线性层
        return x


# 训练模型
def train_model(model, train_loader, val_loader, epochs=300, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            # 将数据移动到 GPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # 将验证数据移动到 GPU
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_outputs = model(X_batch)
                val_loss = criterion(val_outputs, y_batch).item()
                total_val_loss += val_loss

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses


# 主函数
if __name__ == "__main__":
    file_path = "/file_system/vepfs/algorithm/siyu.teng/data/ZhouShan_revised.csv"  # 替换为你的CSV文件路径
    data = load_data(file_path)

    seq_length = 10  # 序列长度
    X, y, scalers = preprocess_data(data, seq_length)

    # 划分训练集、验证集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # 使用 DataLoader
    batch_size = 32  # 设置批量大小
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = 4  # 输入特征数量：COG, SOG, lon, lat
    d_model = 128  # Transformer模型的维度
    nhead = 4  # 多头注意力的头数
    num_layers = 2  # Transformer层数
    output_size = 4  # 输出特征数量：COG, SOG, lon, lat
    model = TransformerModel(input_size, d_model, nhead, num_layers, output_size)

    # 将模型移到 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练模型
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=300, learning_rate=0.001)

    # 测试模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device))

    # 反归一化
    y_test = scalers[list(scalers.keys())[0]].inverse_transform(y_test.cpu().numpy())
    y_pred = scalers[list(scalers.keys())[0]].inverse_transform(y_pred.cpu().numpy())

    # 计算经纬度的均方误差和均方根误差
    lon_mse = np.mean((y_test[:, 2] - y_pred[:, 2]) ** 2)
    lat_mse = np.mean((y_test[:, 3] - y_pred[:, 3]) ** 2)
    lon_rmse = np.sqrt(lon_mse)
    lat_rmse = np.sqrt(lat_mse)

    print(f"Longitude MSE: {lon_mse:.4f}, Longitude RMSE: {lon_rmse:.4f}")
    print(f"Latitude MSE: {lat_mse:.4f}, Latitude RMSE: {lat_rmse:.4f}")

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:, 2], y_test[:, 3], label="True Trajectory")
    plt.plot(y_pred[:, 2], y_pred[:, 3], label="Predicted Trajectory")
    plt.legend()
    plt.show()
