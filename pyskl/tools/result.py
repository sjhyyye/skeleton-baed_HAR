import json
import matplotlib.pyplot as plt
import os

log_file = "work_dirs/posec3d/x3d_shallow_ntu60_xsub/joint/20251118_110929.log.json"
save_dir = "results/posec3d_x3d_ntu60"
os.makedirs(save_dir, exist_ok=True)

# 用字典记录“每个 epoch 最后一条数据”
epoch_data = {}

with open(log_file, "r") as f:
    f.readline()  # 跳过第一行
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except:
            continue

        epoch = data.get("epoch")
        epoch_data[epoch] = data  # 最后一条覆盖前面的

# 将字典按 epoch 排序
epochs = sorted(epoch_data.keys())

loss_list = [epoch_data[e]['loss'] for e in epochs]
top1_list = [epoch_data[e]['top1_acc'] for e in epochs]
top5_list = [epoch_data[e]['top5_acc'] for e in epochs]
lr_list = [epoch_data[e]['lr'] for e in epochs]
grad_list = [epoch_data[e]['grad_norm'] for e in epochs]
memory_list = [epoch_data[e]['memory'] for e in epochs]

def plot_and_save(x, y, title, ylabel, filename):
    plt.figure(figsize=(7,5))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True)

    # ⭐ 找到最高值的位置
    max_idx = y.index(max(y))
    max_x = x[max_idx]
    max_y = y[max_idx]

    # ⭐ 红色标记最高点
    plt.scatter(max_x, max_y, color='red', s=80, zorder=5)

    # ⭐ 标注文字（稍微偏上）
    plt.text(max_x, max_y, f'{max_y:.4f}', color='red', fontsize=10,
             ha='left', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print("Saved:", filename)

plot_and_save(epochs, loss_list, "Loss Curve", "loss", "loss.png")
plot_and_save(epochs, top1_list, "Top1 Accuracy Curve", "top1_acc", "top1_acc.png")
plot_and_save(epochs, top5_list, "Top5 Accuracy Curve", "top5_acc", "top5_acc.png")
plot_and_save(epochs, lr_list, "Learning Rate Curve", "lr", "lr.png")
plot_and_save(epochs, grad_list, "Gradient Norm Curve", "grad_norm", "grad_norm.png")
plot_and_save(epochs, memory_list, "Memory Usage Curve", "memory", "memory.png")
