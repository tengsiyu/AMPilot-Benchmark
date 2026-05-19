import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    # 选择需要的列：MMSI、航向(COG)、速度(SOG)、时间(unixtime)、经度(lon)、纬度(lat)
    data = data.iloc[:, [0, 2, 3, 4, 5, 6]]
    data.columns = ['MMSI', 'Course', 'Speed', 'UnixTime', 'Lon_d', 'Lat_d']
    # 按MMSI和时间排序
    data = data.sort_values(by=['MMSI', 'UnixTime'])
    return data


# 数据预处理
def preprocess_data(data, seq_length):
    # 按MMSI分组处理
    grouped = data.groupby('MMSI')
    X, y, scalers = [], [], {}

    for mmsi, group in grouped:
        # 数据归一化
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(group[['Course', 'Speed', 'Lon_d', 'Lat_d']])
        scalers[mmsi] = scaler

        # 构建序列数据
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:i + seq_length])
            y.append(scaled_data[i + seq_length])

    X = np.array(X)
    y = np.array(y)

    # 转换为PyTorch张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y, scalers


# 构建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取序列的最后一个时间步的输出
        return out


# 训练模型
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


# 主函数
if __name__ == "__main__":
    # 加载数据
    file_path = '/file_system/vepfs/algorithm/siyu.teng/data/ZhouShan_revised.csv'  # 替换为你的CSV文件路径
    data = load_data(file_path)

    # 数据预处理
    seq_length = 10  # 序列长度
    X, y, scalers = preprocess_data(data, seq_length)

    # 划分训练集、验证集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # 初始化模型
    input_size = 4  # 输入特征数量：COG, SOG, lon, lat
    hidden_size = 50
    output_size = 4  # 输出特征数量：COG, SOG, lon, lat
    num_layers = 2
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)

    # 训练模型
    train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.001)

    # 测试模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    # 反归一化
    y_test = scalers[list(scalers.keys())[0]].inverse_transform(y_test.numpy())
    y_pred = scalers[list(scalers.keys())[0]].inverse_transform(y_pred.numpy())

    # 计算经纬度的均方误差和均方根误差
    lon_mse = np.mean((y_test[:, 2] - y_pred[:, 2]) ** 2)
    lat_mse = np.mean((y_test[:, 3] - y_pred[:, 3]) ** 2)
    lon_rmse = np.sqrt(lon_mse)
    lat_rmse = np.sqrt(lat_mse)

    print(f'Longitude MSE: {lon_mse:.4f}, Longitude RMSE: {lon_rmse:.4f}')
    print(f'Latitude MSE: {lat_mse:.4f}, Latitude RMSE: {lat_rmse:.4f}')

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:, 2], y_test[:, 3], label='True Trajectory')
    plt.plot(y_pred[:, 2], y_pred[:, 3], label='Predicted Trajectory')
    plt.legend()
    plt.show()

    # 可视化训练损失和验证损失
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()