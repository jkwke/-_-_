import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# ================= 1. 数据解析与预处理 =================

def parse_game_string(game_str):

    """解析单局棋谱字符串，返回坐标列表 [(row, col), ...]"""
    moves_str = re.findall(r'([a-o]\d+)', game_str)
    moves = []
    for move in moves_str:
        col = ord(move[0]) - ord('a')  # a-o -> 0-14
        row = int(move[1:]) - 1  # 1-15 -> 0-14
        if 0 <= row < 15 and 0 <= col < 15:
            moves.append((row, col))
    return moves


class GomokuDataset(Dataset):
    """五子棋模仿学习数据集"""

    def __init__(self, folder_path):
        self.states = []
        self.actions = []

        # 读取文件夹中所有.txt文件
        all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        content = []
        for file in all_files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content.append(f.read())
        merged_content = ' '.join(content)  # 合并所有文件内容

        # 按空格切分多局棋谱
        games = merged_content.split()  # ["a1b2c3", "d4e5f6", "g7h8i9"]

        for game_str in games:
            moves = parse_game_string(game_str)  # game_str -> "a1b2c3" -> [(0, 0), (1, 1), (2, 2)]
            # 初始化 15x15 棋盘 (通道0: 黑棋, 通道1: 白棋)
            board = np.zeros((2, 15, 15), dtype=np.float32)

            for i, (r, c) in enumerate(moves):
                current_player = i % 2  # 0: 黑棋, 1: 白棋

                # 构造模型输入 (3通道)
                state = np.zeros((3, 15, 15), dtype=np.float32)
                if current_player == 0:  # 黑棋下
                    state[0] = board[0]  # 当前=黑棋
                    state[1] = board[1]  # 对手=白棋
                    state[2, :, :] = 1.0  # 标识位=1
                else:  # 白棋下
                    state[0] = board[1]  # 当前=白棋
                    state[1] = board[0]  # 对手=黑棋
                    state[2, :, :] = 0.0  # 标识位=0

                # 记录状态-动作对
                self.states.append(state)
                action_idx = r * 15 + c  # 将2D坐标展平为 0-224 的索引
                self.actions.append(action_idx)

                # 在棋盘上落子，为下一步做准备
                board[current_player, r, c] = 1.0

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.tensor(self.states[idx]), torch.tensor(self.actions[idx], dtype=torch.long)


# ================= 2. 神经网络模型 =================

class ResidualBlock(nn.Module):
    """残差块：两层卷积 + 跳跃连接"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    """前面的conv和bn就是卷积核提取特征后作为残差，和下面的identity（原始数据）相加后共同作为结果传出，以实现
     数据信息不丢失"""
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # 残差连接
        out = self.relu(out)
        return out


class GomokuNet(nn.Module):
    """五子棋残差卷积网络"""

    def __init__(self, num_res_blocks=6):
        super(GomokuNet, self).__init__()

        # 输入层：3通道 -> 64通道
        self.input_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(64)

        # 堆叠多个残差块，逐步加深网络
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(num_res_blocks)
        ])

        # 策略头：全局平均池化 + 全连接
        # (B, 64, 15, 15) -> GAP -> (B, 64) -> FC -> (B, 225)
        self.policy_fc = nn.Linear(64, 225)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_bn(self.input_conv(x)))

        for res_block in self.res_blocks:
            x = res_block(x)

        # 全局平均池化，保留通道维度，压平空间维度
        x = x.mean(dim=[2, 3])  # (B, 64, 15, 15) -> (B, 64)
        x = self.policy_fc(x)   # (B, 64) -> (B, 225)
        return x


# ================= 3. 训练循环 =================

def evaluate(model, dataloader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for states, actions in dataloader:
            states, actions = states.to(device), actions.to(device)
            outputs = model(states)
            loss = criterion(outputs, actions)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += actions.size(0)
            correct_preds += (predicted == actions).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct_preds / total_samples
    return avg_loss, accuracy


def train_model(folder_path, epochs=30, batch_size=64, model_save_path="gomoku_model_30.pth", val_split=0.1):
    # 检查GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载全量数据
    full_dataset = GomokuDataset(folder_path)
    print(f"训练样本总数: {len(full_dataset)}")

    # 划分训练集和验证集
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"训练集: {train_size}, 验证集: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、优化器、损失函数
    model = GomokuNet(num_res_blocks=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        total_loss = 0
        correct_preds = 0
        total_samples = 0

        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)

            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += actions.size(0)
            correct_preds += (predicted == actions).sum().item()

        scheduler.step()  # 更新学习率

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct_preds / total_samples

        # --- 验证阶段 ---
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path.replace('.pth', '_best.pth'))

        print(f"轮次 [{epoch + 1}/{epochs}], "
              f"训练损失: {train_loss:.4f}, 训练准确度: {train_acc:.2f}%, "
              f"验证损失: {val_loss:.4f}, 验证准确度: {val_acc:.2f}%")

    # 保存最终模型
    torch.save(model.state_dict(), model_save_path)
    print(f"\n训练完成！最佳验证准确度: {best_val_acc:.2f}%")
    print(f"最终模型保存在 {model_save_path}")
    print(f"最佳模型保存在 {model_save_path.replace('.pth', '_best.pth')}")


# ================= 4. 主程序 =================
if __name__ == "__main__":
    # 训练参数
    TRAIN_FOLDER = r"D:\quarkdownload\train_data_2"
    EPOCHS = 40   # 训练40轮
    BATCH_SIZE = 64
    MODEL_SAVE_PATH = "gomoku_model_30.pth"

    # 开始训练
    train_model(
        folder_path=TRAIN_FOLDER,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_path=MODEL_SAVE_PATH,
        val_split=0.1  # 10% 用于验证
    )
