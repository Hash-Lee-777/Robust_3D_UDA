import matplotlib.pyplot as plt
import numpy as np

# 攻击后的点数
attack_after_counts = {
    0: 9963553,
    1: 5508098,
    2: 4693629,
    3: 16750548,
    4: 18862278,
    5: 7916191,
    6: 5708065,
    7: 6772297,
    8: 252306832,
    9: 3662696,
    10: 111960530,
    11: 8195591,
    12: 169462040,
    13: 31916846,
    14: 264528127,
    15: 16276234,
    16: 275604758,
    17: 9632596,
    18: 4659534
}

# 攻击前的点数
attack_before_counts = {
    0: 14246691,
    1: 1245907,
    2: 3158632,
    3: 14871829,
    4: 13994321,
    5: 6425325,
    6: 6402611,
    7: 8767859,
    8: 437089410,
    9: 3120092,
    10: 150828121,
    11: 6371882,
    12: 218063916,
    13: 33760023,
    14: 99866964,
    15: 11052985,
    16: 125973501,
    17: 11877867,
    18: 2171905
}

# 创建类别序号和对应点数
categories = list(range(19))
attack_after_values = [attack_after_counts.get(cat, 0) for cat in categories]
attack_before_values = [attack_before_counts.get(cat, 0) for cat in categories]

# 绘制柱状图
x = np.arange(len(categories))  # 类别序号
width = 0.35  # 柱宽

fig, ax = plt.subplots(figsize=(14, 8))
bars1 = ax.bar(x - width/2, attack_before_values, width, label='Attack Before', color='#1f77b4')
bars2 = ax.bar(x + width/2, attack_after_values, width, label='Attack After', color='#ff7f0e')

# 添加标签和标题
ax.set_xlabel('Category')
ax.set_ylabel('Number of Points')
ax.set_title('Number of Points per Category Before and After Attack')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# 显示图形
plt.tight_layout()
plt.show()
