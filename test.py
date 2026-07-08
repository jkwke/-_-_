import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn  # 正确：导入 PyTorch 的神经网络模块，并命名为 nn
import numpy as np


# ================= 模型定义（需与训练时一致） =================

class ResidualBlock(nn.Module):
    """残差块：两层卷积 + 跳跃连接"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class GomokuNet(nn.Module):
    """五子棋残差卷积网络"""

    def __init__(self, num_res_blocks=6):
        super(GomokuNet, self).__init__()

        # 输入层：3通道 -> 64通道
        self.input_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(64)

        # 堆叠多个残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(num_res_blocks)
        ])

        # 策略头：全局平均池化 + 全连接
        self.policy_fc = nn.Linear(64, 225)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_bn(self.input_conv(x)))

        for res_block in self.res_blocks:
            x = res_block(x)

        # 全局平均池化
        x = x.mean(dim=[2, 3])  # (B, 64, 15, 15) -> (B, 64)
        x = self.policy_fc(x)   # (B, 64) -> (B, 225)
        return x


# ================= 加载模型函数 =================
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GomokuNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# ================= 规则检查辅助函数 =================

def check_five_at_position(board_matrix, row, col, player):
    """检查在 (row, col) 落子后是否形成五连珠"""
    board_matrix[row, col] = player  # 临时落子
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < 15 and 0 <= c < 15 and board_matrix[r, c] == player:
            count += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < 15 and 0 <= c < 15 and board_matrix[r, c] == player:
            count += 1
            r -= dr
            c -= dc
        if count >= 5:
            board_matrix[row, col] = 0  # 恢复棋盘
            return True
    board_matrix[row, col] = 0  # 恢复棋盘
    return False


def find_immediate_win(board_matrix, player):
    """查找当前玩家是否有一步制胜的位置"""
    for r in range(15):
        for c in range(15):
            if board_matrix[r, c] == 0:
                if check_five_at_position(board_matrix, r, c, player):
                    return r, c
    return None


def find_immediate_block(board_matrix, opponent_player):
    """查找是否需要堵住对手的一步制胜"""
    for r in range(15):
        for c in range(15):
            if board_matrix[r, c] == 0:
                if check_five_at_position(board_matrix, r, c, opponent_player):
                    return r, c
    return None


# ================= AI预测函数（增强版） =================

def predict_move(model, board_matrix, current_player_is_black):
    """
    使用训练好的模型预测下一步
    board_matrix: 15x15 的 numpy 数组, 0=空, 1=黑, 2=白
    current_player_is_black: True=当前黑棋下, False=当前白棋下
    """
    current_player = 1 if current_player_is_black else 2
    opponent_player = 2 if current_player_is_black else 1

    # ---- 规则检查 1：自己能否一步获胜 ----
    win_move = find_immediate_win(board_matrix, current_player)
    if win_move is not None:
        return win_move

    # ---- 规则检查 2：对手能否一步获胜（需要堵住） ----
    block_move = find_immediate_block(board_matrix, opponent_player)
    if block_move is not None:
        return block_move

    # ---- 模型推理（原逻辑） ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    state = np.zeros((3, 15, 15), dtype=np.float32)
    black_board = (board_matrix == 1).astype(np.float32)
    white_board = (board_matrix == 2).astype(np.float32)

    if current_player_is_black:
        state[0] = black_board
        state[1] = white_board
        state[2, :, :] = 1.0
    else:
        state[0] = white_board
        state[1] = black_board
        state[2, :, :] = 0.0

    with torch.no_grad():
        input_tensor = torch.tensor(state).unsqueeze(0).to(device)
        logits = model(input_tensor).squeeze(0).cpu().numpy()

    for r in range(15):
        for c in range(15):
            if board_matrix[r, c] != 0:
                logits[r * 15 + c] = -float('inf')

    best_action_idx = np.argmax(logits)
    best_row = best_action_idx // 15
    best_col = best_action_idx % 15

    return best_row, best_col


# ================= 五子棋游戏类（集成AI） =================
class Gomoku:
    def __init__(self):
        self.board_size = 15
        self.cell_size = 35
        self.piece_radius = 15
        # 初始化棋盘 0=空, 1=黑棋(玩家), 2=白棋(AI)
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        # 当前玩家 1=黑棋(玩家), 2=白棋(AI)
        self.current_player = 1
        # 游戏是否结束
        self.game_over = False
        # 加载训练好的模型
        self.model = load_model("gomoku_model_best_40.pth")  # 确保模型路径正确
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("五子棋 - 玩家 vs AI")
        self.root.resizable(False, False)
        # 创建画布
        canvas_size = self.board_size * self.cell_size + 40
        self.canvas = tk.Canvas(
            self.root, width=canvas_size, height=canvas_size, bg="#DEB887"
        )
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_click)
        # 状态标签
        self.status_var = tk.StringVar(value="黑棋回合（玩家）")
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var, font=("Arial", 14), pady=10
        )
        self.status_label.pack()
        # 重置按钮
        reset_btn = tk.Button(
            self.root, text="重新开始", command=self.reset_game, font=("Arial", 12), padx=20
        )
        reset_btn.pack(pady=5)
        # 绘制棋盘
        self.draw_board()

    def draw_board(self):
        """绘制棋盘网格"""
        offset = 20
        for i in range(self.board_size):
            # 绘制横线
            y = offset + i * self.cell_size
            self.canvas.create_line(
                offset, y, offset + (self.board_size - 1) * self.cell_size, y, fill="black", width=1
            )
            # 绘制竖线
            x = offset + i * self.cell_size
            self.canvas.create_line(
                x, offset, x, offset + (self.board_size - 1) * self.cell_size, fill="black", width=1
            )
        # 重新绘制所有棋子
        self.draw_all_pieces()

    def draw_all_pieces(self):
        """绘制所有棋子"""
        offset = 20
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] != 0:
                    x = offset + col * self.cell_size
                    y = offset + row * self.cell_size
                    color = "black" if self.board[row][col] == 1 else "white"
                    self.canvas.create_oval(
                        x - self.piece_radius, y - self.piece_radius,
                        x + self.piece_radius, y + self.piece_radius,
                        fill=color, outline="black", width=2
                    )

    def on_click(self, event):
        """处理玩家（黑棋）的点击事件"""
        if self.game_over or self.current_player != 1:
            return
        offset = 20
        # 计算点击的格子坐标
        col = round((event.x - offset) / self.cell_size)
        row = round((event.y - offset) / self.cell_size)
        # 检查边界
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return
        # 检查该位置是否已有棋子
        if self.board[row][col] != 0:
            return
        # 玩家落子
        self.board[row][col] = self.current_player
        # 重新绘制棋盘
        self.canvas.delete("all")
        self.draw_board()
        # 检查是否获胜
        if self.check_win(row, col):
            winner = "黑棋（玩家）" if self.current_player == 1 else "白棋（AI）"
            self.status_var.set(f"{winner}获胜！")
            self.game_over = True
            messagebox.showinfo("游戏结束", f"{winner}获胜！")
            return
        # 检查是否平局
        if self.check_draw():
            self.status_var.set("平局！")
            self.game_over = True
            messagebox.showinfo("游戏结束", "平局！")
            return
        # 切换到AI回合
        self.current_player = 2
        self.status_var.set("白棋回合（AI）")
        # 延迟500ms让AI下棋（避免界面卡顿）
        self.root.after(500, self.ai_move)

    def ai_move(self):
        """AI（白棋）自动下棋"""
        if self.game_over or self.current_player != 2:
            return
        # 获取当前棋盘状态
        board_matrix = np.array(self.board)
        # AI下棋（白棋，current_player_is_black=False）
        row, col = predict_move(self.model, board_matrix, current_player_is_black=False)
        # 防止模型预测非法位置（fallback：随机选空位）
        if self.board[row][col] != 0:
            empty_positions = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if
                               self.board[r][c] == 0]
            if empty_positions:
                row, col = empty_positions[np.random.choice(len(empty_positions))]
        # AI落子
        self.board[row][col] = self.current_player
        # 重新绘制棋盘
        self.canvas.delete("all")
        self.draw_board()
        # 检查是否获胜
        if self.check_win(row, col):
            winner = "白棋（AI）" if self.current_player == 2 else "黑棋（玩家）"
            self.status_var.set(f"{winner}获胜！")
            self.game_over = True
            messagebox.showinfo("游戏结束", f"{winner}获胜！")
            return
        # 检查是否平局
        if self.check_draw():
            self.status_var.set("平局！")
            self.game_over = True
            messagebox.showinfo("游戏结束", "平局！")
            return
        # 切换回玩家回合
        self.current_player = 1
        self.status_var.set("黑棋回合（玩家）")

    def check_win(self, row, col):
        """检查是否在(row, col)位置形成五连珠"""
        player = self.board[row][col]
        directions = [
            (0, 1),  # 水平
            (1, 0),  # 垂直
            (1, 1),  # 对角线
            (1, -1)  # 反对角线
        ]
        for dr, dc in directions:
            count = 1  # 包含当前位置
            # 正向检查
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc
            # 反向检查
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 5:
                return True
        return False

    def check_draw(self):
        """检查是否平局（棋盘已满）"""
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == 0:
                    return False
        return True

    def reset_game(self):
        """重置游戏"""
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = 1
        self.game_over = False
        self.status_var.set("黑棋回合（玩家）")
        self.canvas.delete("all")
        self.draw_board()

    def run(self):
        """运行游戏"""
        self.root.mainloop()


if __name__ == "__main__":
    game = Gomoku()
    game.run()
