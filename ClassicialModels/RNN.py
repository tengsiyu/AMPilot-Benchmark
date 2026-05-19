import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# 1. 数据加载和预处理
def load_and_preprocess_data(file_path):
    # 加载CSV数据
    data = pd.read_csv(file_path)

    # 提取需要的列：MMSI, COG, Speed, Unixtime, lon, lat
    data = data.iloc[:, [0, 2, 3, 4, 5, 6]]
    data.columns = ['MMSI', 'COG', 'SOG', 'Unixtime', 'lon', 'lat']

    # 按MMSI和时间排序
    data.sort_values(by=['MMSI', 'Unixtime'], inplace=True)

    # 将时间戳转换为时间差（秒）
    data['TimeDiff'] = data.groupby('MMSI')['Unixtime'].diff().fillna(0)

    # 选择一个MMSI进行建模（这里选择第一个MMSI）
    selected_mmsi = data['MMSI'].iloc[0]
    data = data[data['MMSI'] == selected_mmsi].reset_index(drop=True)

    return data


# 2. 数据归一化
def normalize_data(data):
    scaler = MinMaxScaler()
    data[['lon', 'lat']] = scaler.fit_transform(data[['lon', 'lat']])
    return data, scaler


# 3. 创建时间序列数据
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[['lon', 'lat']].iloc[i:i + seq_length].values)
        y.append(data[['lon', 'lat']].iloc[i + seq_length].values)
    return np.array(X), np.array(y)


# 4. 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


# 5. 训练和验证模型
def train_and_validate_model(model, X_train, y_train, X_val, y_val, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        train_loss = criterion(output, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}')


# 6. 测试模型
def test_model(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        output = scaler.inverse_transform(output.numpy())
        y_test = scaler.inverse_transform(y_test.numpy())

        # 计算RMSE
        rmse_lon = np.sqrt(mean_squared_error(y_test[:, 0], output[:, 0]))
        rmse_lat = np.sqrt(mean_squared_error(y_test[:, 1], output[:, 1]))
        print(f'RMSE for Longitude: {rmse_lon:.6f}')
        print(f'RMSE for Latitude: {rmse_lat:.6f}')

        # 绘制预测结果
        plt.figure(figsize=(12, 6))
        plt.plot(y_test[:, 0], y_test[:, 1], label='True')
        plt.plot(output[:, 0], output[:, 1], label='Predicted')
        plt.legend()
        plt.title('True vs Predicted Trajectory')
        plt.show()


# 7. 主函数
def main(file_path):
    # 加载和预处理数据
    data = load_and_preprocess_data(file_path)
    data, scaler = normalize_data(data)

    # 创建序列数据
    seq_length = 5
    X, y = create_sequences(data, seq_length)

    # 划分训练集、验证集和测试集
    train_size = int(len(X) * 0.8)
    val_size = int(len(X) * 0.1)
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

    # 转换为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 定义模型
    input_size = 2
    hidden_size = 50
    output_size = 2
    num_layers = 1
    model = RNNModel(input_size, hidden_size, output_size, num_layers)

    # 训练和验证模型
    epochs = 100
    learning_rate = 0.001
    train_and_validate_model(model, X_train, y_train, X_val, y_val, epochs, learning_rate)

    # 测试模型
    test_model(model, X_test, y_test, scaler)


# 运行主函数
if __name__ == "__main__":
    file_path = '/file_system/vepfs/algorithm/siyu.teng/data/ZhouShan_revised.csv'  # 替换为你的CSV文件路径
    main(file_path)