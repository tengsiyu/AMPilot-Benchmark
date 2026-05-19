import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 加载 JSON 数据
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

    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = torch.tensor(outputs, dtype=torch.float32)

    all_coords = torch.cat([inputs.reshape(-1, 2), outputs.reshape(-1, 2)], dim=0).numpy()
    scaler = MinMaxScaler()
    scaler.fit(all_coords)

    inputs = torch.tensor(scaler.transform(inputs.reshape(-1, 2)).reshape(inputs.shape), dtype=torch.float32)
    outputs = torch.tensor(scaler.transform(outputs.reshape(-1, 2)).reshape(outputs.shape), dtype=torch.float32)

    return inputs.to(DEVICE), outputs.to(DEVICE), scaler


# 1.b 仅解析未缩放的数据（CPU 张量），用于分离的训练/测试加载
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

    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(outputs, dtype=torch.float32),
    )


# 1.c 对 (X, y) 同步缩放；若未提供 scaler，则由 (X, y) 拟合
def scale_pairs(X, Y, scaler: MinMaxScaler = None):
    if scaler is None:
        all_coords = torch.cat([X.reshape(-1, 2), Y.reshape(-1, 2)], dim=0).numpy()
        scaler = MinMaxScaler().fit(all_coords)
    Xs = torch.tensor(scaler.transform(X.reshape(-1, 2)).reshape(X.shape), dtype=torch.float32)
    Ys = torch.tensor(scaler.transform(Y.reshape(-1, 2)).reshape(Y.shape), dtype=torch.float32)
    return Xs, Ys, scaler


# 2. 图卷积层
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        B, T, N, C = x.shape
        x = x.view(B * T, N, C)
        adj = adj.repeat_interleave(T, dim=0)
        x = torch.bmm(adj, x)
        x = self.fc(x)
        return x.view(B, T, N, -1)


# 3. ST-GCN 模型
class STGCNModel(nn.Module):
    def __init__(self, node_features=2, hidden_dim=64, output_steps=6):
        super(STGCNModel, self).__init__()
        self.gc1 = GraphConvolution(node_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)

        self.temporal_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
        self.temporal_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, output_steps * 2)
        self.output_steps = output_steps

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))

        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.temporal_conv1(x))
        x = F.relu(self.temporal_conv2(x))

        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x.view(-1, self.output_steps, 2)


# 4. 辅助函数


def wrap_single_ship_input(x_seq):
    return x_seq.unsqueeze(2)


def create_identity_adj(batch_size, num_nodes=1):
    return torch.eye(num_nodes, device=DEVICE).unsqueeze(0).repeat(batch_size, 1, 1)


# 5. 训练模型


def train_model(model, X_train, y_train, epochs=100, learning_rate=0.001, batch_size=512):
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            x_input = wrap_single_ship_input(xb)
            adj = create_identity_adj(xb.size(0))

            optimizer.zero_grad()
            with autocast():
                preds = model(x_input, adj)
                loss = criterion(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(loader):.6f}")


# 6. 测试并保存预测


def test_model(model, X_test, y_test, scaler, save_path="Output/STGCN_predictions.json"):
    model.eval()
    predictions = []

    with torch.no_grad():
        x_input = wrap_single_ship_input(X_test)
        adj = create_identity_adj(X_test.size(0))

        output = model(x_input, adj)

        output_np = output.cpu().numpy().reshape(-1, 2)
        y_test_np = y_test.cpu().numpy().reshape(-1, 2)

        output_np = scaler.inverse_transform(output_np)
        y_test_np = scaler.inverse_transform(y_test_np)

        output_np = output_np.reshape(-1, 6, 2)
        y_test_np = y_test_np.reshape(-1, 6, 2)

        for pred, label in zip(output_np, y_test_np):
            predictions.append({"prediction": pred.tolist(), "label": label.tolist()})

        with open(save_path, "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"\n✅ Predictions saved to: {save_path}")

        rmse_lon = np.sqrt(mean_squared_error(y_test_np[:, :, 0].flatten(), output_np[:, :, 0].flatten()))
        rmse_lat = np.sqrt(mean_squared_error(y_test_np[:, :, 1].flatten(), output_np[:, :, 1].flatten()))
        print(f"📈 RMSE for Longitude: {rmse_lon:.6f}")
        print(f"📈 RMSE for Latitude:  {rmse_lat:.6f}")


# 7. 主函数入口


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

    model = STGCNModel(node_features=2, hidden_dim=64, output_steps=6).to(DEVICE)
    model = torch.compile(model)

    train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, batch_size=512)

    test_model(
        model,
        X_test,
        y_test,
        scaler,
        # save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/ST_GCN_Zhoushan_predictions_6outputs.json",
        # save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/ST_GCN_EMS_predictions_6outputs.json",
        # save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/ST_GCN_SG_predictions_6outputs.json",
        # save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/ST_GCN_predictions_0_5draught.json",
        # save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/ST_GCN_predictions_5_10draught.json",
        save_path="/file_system/vepfs/algorithm/siyu.teng/Classical_Model/Output/ST_GCN_predictions_10_25draught.json",
    )


# 新增：主流程（加载训练集→训练→保存模型→加载测试集→测试）
def main_separate_datasets(train_file_path,
                           test_file_path,
                           model_save_path="Output/STGCN_Local_Model.pth",
                           pred_save_path="Output/STGCN_predictions.json"):
    np.random.seed(42)
    torch.manual_seed(42)

    # 1) 训练集：解析未缩放
    X_train_raw, y_train_raw = parse_json_as_tensor_pairs(train_file_path)
    print(f"✅ 训练集帧数: {len(X_train_raw)}")

    # 2) 用训练集拟合 scaler 并缩放
    X_train, y_train, scaler = scale_pairs(X_train_raw, y_train_raw, scaler=None)
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)

    # 3) 构建并训练模型
    model = STGCNModel(node_features=2, hidden_dim=64, output_steps=6).to(DEVICE)
    model = torch.compile(model)
    train_model(model, X_train, y_train, epochs=300, learning_rate=0.001, batch_size=512)

    # 4) 保存模型
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "node_features": 2,
                "hidden_dim": 64,
                "output_steps": 6,
            }
        }, model_save_path)
        print(f"✅ Model saved to: {model_save_path}")

    # 5) 测试集：解析未缩放并用训练集 scaler 缩放
    X_test_raw, y_test_raw = parse_json_as_tensor_pairs(test_file_path)
    print(f"✅ 测试集帧数: {len(X_test_raw)}")
    X_test, y_test, _ = scale_pairs(X_test_raw, y_test_raw, scaler=scaler)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    # 6) 测试与保存预测
    if pred_save_path:
        os.makedirs(os.path.dirname(pred_save_path) or ".", exist_ok=True)
    test_model(model, X_test, y_test, scaler, save_path=pred_save_path)


# 8. 启动运行
if __name__ == "__main__":
    # 按需修改以下路径
    train_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/draught_5_10m_6outputs.json"
    test_file = "/file_system/vepfs/algorithm/siyu.teng/MarineData/draught_10_25m_6outputs.json"

    model_path = "Output/STGCN_Local_Model.pth"
    pred_path = "Output/STGCN_predictions.json"

    main_separate_datasets(train_file, test_file, model_path, pred_path)
