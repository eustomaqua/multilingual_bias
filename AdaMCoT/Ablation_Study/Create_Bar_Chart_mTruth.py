
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import matplotlib.gridspec as gridspec
# --------------------------------------------------------------------------
# 1. Data Parsing (No changes here)
# --------------------------------------------------------------------------
def parse_line(line):
    """Parses a single line of text data into a structured dictionary."""
    if not line:
        return {'total': 0, 'breakdown': {}}
    
    parts = line.split(" ", 1)
    total = int(parts[0])
    breakdown_str = parts[1]
    
    breakdown = {}
    matches = re.findall(r'([A-Za-z]+):\s*([\d\.]+)%', breakdown_str)
    for match in matches:
        category, percent = match
        breakdown[category.capitalize()] = float(percent)
    
    current_total_percent = sum(breakdown.values())
    if current_total_percent > 0:
        factor = 100.0 / current_total_percent
        for cat in breakdown:
            breakdown[cat] *= factor
            
    return {'total': total, 'breakdown': breakdown}

w_to_c_raw = [
    "101 English: 81.19% directly: 15.84% Indonesian: 2.97%", "170 directly: 38.24% English: 61.18% Bengali: 0.59%",
    "87 Chinese: 21.84% English: 37.93% directly: 37.93% Indonesian: 2.30%", "87 Chinese: 66.67% English: 5.75% directly: 27.58%", "82 English: 36.59% Chinese: 47.56% Indonesian: 15.85%",
    "56 English: 55.36% Chinese: 17.86% directly: 26.78%", "182 English: 50.00% directly: 50.00%",
    "90 English: 47.78% Indonesian: 8.89% directly: 43.33%", "111 Chinese: 36.94% English: 46.85% directly: 16.21%",
    "108 English: 93.52% directly: 5.56% Chinese: 0.92%", "126 English: 33.33% Indonesian: 5.56% Chinese: 61.11%",
    "214 Chinese: 6.07% directly: 64.49% Indonesian: 29.44%", "95 Indonesian: 12.63% directly: 55.79% English: 31.58%",
    "144 English: 87.50% directly: 12.50%", "62 Indonesian: 51.61% English: 30.65% Chinese: 17.74%",
    "89 English: 23.60% Chinese: 52.81% directly: 23.59%", "123 Chinese: 51.22% English: 28.46% directly: 20.32%",
    "120 Marathi: 20.83% English: 70.00% directly: 9.17%", "128 directly: 10.94% English: 84.38% Nepali: 4.68%",
    "101 English: 19.80% Chinese: 60.40% directly: 19.80%", "150 directly: 78.67% English: 9.33% Chinese: 12.00%",
    "83 directly: 32.53% Indonesian: 44.58% Chinese: 22.89%", "120 directly: 16.67% Indonesian: 41.67% English: 41.66%",
    "133 directly: 49.62% English: 48.87% Indonesian: 1.51%", "126 directly: 34.13% English: 27.78% Serbian: 38.09%",
    "92 English: 27.17% directly: 27.17% Chinese: 45.66%", "106 directly: 46.23% English: 35.85% Chinese: 17.92%",
    "135 Chinese: 20.00% Indonesian: 2.22% directly: 77.78%", "129 Indonesian: 31.01% Chinese: 14.73% directly: 54.26%",
    "120 English: 69.17% directly: 30.83%", "100 English: 84.00% directly: 16.00%",
    "67 Chinese: 22.39% directly: 65.67% Indonesian: 11.94%",
]

c_to_w_raw = [
    "43 English: 86.05% directly: 9.30% Indonesian: 4.65%", "64 English: 79.69% directly: 18.75% Chinese: 1.56%",
    "48 Chinese: 41.67% directly: 25.00% English: 33.33%", 
    "72 Chinese: 45.83% directly: 54.17%", "38 Chinese: 55.26% English: 15.79% Indonesian: 28.95%",
    "32 English: 46.88% directly: 21.88% Chinese: 31.24%", "59 English: 52.54% directly: 47.46%",
    "49 Indonesian: 22.45% directly: 14.29% English: 63.26%", "107 directly: 1.87% English: 66.36% Chinese: 31.77%",
    "38 English: 100.00%", "65 Indonesian: 6.15% English: 40.00% directly: 53.85%",
    "43 Indonesian: 44.19% Chinese: 32.56% English: 23.25%", "71 Chinese: 28.17% Armenian: 1.41% Indonesian: 70.42%",
    "45 English: 84.44% Chinese: 4.44% directly: 11.12%", "39 directly: 15.38% English: 17.95% Indonesian: 66.67%",
    "84 Chinese: 61.90% English: 25.00% directly: 13.10%", "120 Chinese: 59.17% English: 33.33% directly: 7.50%",
    "100 English: 88.00% Chinese: 7.00% Marathi: 5.00%", "63 English: 96.83% Chinese: 1.59% directly: 1.58%",
    "49 Chinese: 67.35% directly: 10.20% English: 22.45%", "43 Chinese: 11.63% English: 39.53% Indonesian: 48.84%",
    "54 directly: 18.52% Indonesian: 38.89% Chinese: 42.59%", "22 English: 50.00% Indonesian: 36.36% directly: 13.64%",
    "47 English: 76.60% directly: 14.89% Indonesian: 8.51%", "65 English: 36.92% Chinese: 23.08% directly: 40.00%",
    "68 English: 23.53% Indonesian: 22.06% Chinese: 54.41%", "134 English: 71.64% directly: 6.72% Chinese: 21.64%",
    "114 directly: 11.40% English: 61.40% Indonesian: 27.20%", "56 directly: 30.36% Chinese: 14.29% Indonesian: 55.35%",
    "49 English: 61.22% directly: 38.78%", "29 English: 86.21% directly: 13.79%",
    "50 Chinese: 28.00% directly: 48.00% Indonesian: 24.00%",
]



language_order = ['ar', 'bn', 'ca', 'da', 'de', 'es', 'eu', 'fr', 'gu', 'hi', 'hr',
                  'hu', 'hy', 'id', 'it', 'kn', 'ml', 'mr', 'ne', 'nl', 'pt', 'ro',
                  'ru', 'sk', 'sr', 'sv', 'ta', 'te', 'uk', 'vi', 'zh', "en"]

w_to_c_parsed = [parse_line(line) for line in w_to_c_raw]; w_to_c_parsed.append({'total': 0, 'breakdown': {}})
c_to_w_parsed = [parse_line(line) for line in c_to_w_raw]
mtruth_data = {'From Wrong to Right': dict(zip(language_order, w_to_c_parsed)), 'From Correct to Wrong': dict(zip(language_order, c_to_w_parsed))}
all_cats_raw = set(); [all_cats_raw.add(cat) for d, l in mtruth_data.items() for v in l.values() for cat in v['breakdown'].keys()]
categories = sorted([f'{cat} Thought' if cat != 'Directly' else 'Directly Answer' for cat in all_cats_raw])
records = []; [records.append({'Language': l, 'Direction': d, 'Category': f"{c} Thought" if c != 'Directly' else 'Directly Answer', 'Count': v['total'] * (p/100.0)}) for d, ld in mtruth_data.items() for l,v in ld.items() if v['total'] > 0 for c, p in v['breakdown'].items()]
df = pd.DataFrame(records)

# --------------------------------------------------------------------------
# 3. Plotting
# --------------------------------------------------------------------------
directions = ['From Wrong to Right', 'From Correct to Wrong']
direction_labels = {'From Wrong to Right': 'Improvement (Wrong to Correct)', 'From Correct to Wrong': 'Regression (Correct to Wrong)'}
sns.set_theme(style="whitegrid", font_scale=1.2)
palette = sns.color_palette("tab20", len(categories))
# category_colors = dict(zip(categories, palette))
categories=['Armenian Thought', 'Bengali Thought', 'Chinese Thought', 'Directly Answer', 'English Thought', 'Indonesian Thought', 'Marathi Thought', 'Nepali Thought', 'Serbian Thought']
palette = [(0.17254901960784313, 0.6274509803921569, 0.17254901960784313), (1.0, 0.7333333333333333, 0.47058823529411764), (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (0.6823529411764706, 0.7803921568627451, 0.9098039215686274), (1.0, 0.4980392156862745, 0.054901960784313725), (0.596078431372549, 0.8745098039215686, 0.5411764705882353), (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), (1.0, 0.596078431372549, 0.5882352941176471), (0.5803921568627451, 0.403921568627451, 0.7411764705882353)]
# print(categories)
# print(palette)

category_colors = dict(zip(categories, palette))
fig = plt.figure(figsize=(20, 13))

# 使用 GridSpec 进行布局
# hspace: 进一步减小，让子图更近
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2], hspace=0.15) 
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
axes = [ax1, ax2]

for row_idx, direction in enumerate(directions):
    ax = axes[row_idx]
    plot_df = df[df['Direction'] == direction]
    pivot = plot_df.pivot_table(index='Language', columns='Category', values='Count', fill_value=0, aggfunc='sum').reindex(index=language_order, columns=categories, fill_value=0)

    bottom = np.zeros(len(language_order))
    for cat in categories:
        if cat in pivot.columns:
            ax.bar(language_order, pivot[cat], bottom=bottom, color=category_colors[cat], label=cat)
            bottom += pivot[cat].values

    # 独立计算和设置 Y 轴范围
    if not plot_df.empty:
        max_y_for_this_plot = plot_df.groupby('Language')['Count'].sum().max()
        ax.set_ylim(0, max_y_for_this_plot * 1.05) # 稍微增加顶部空间给数字标签
    else:
        ax.set_ylim(0, 10)

    text_padding = ax.get_ylim()[1] * 0.02
    for idx, lang in enumerate(language_order):
        total = pivot.loc[lang].sum()
        if total > 0:
            ax.text(idx, total + text_padding, f'{total:.0f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax.set_title(f'{direction_labels[direction]}', fontsize=20, fontweight='bold', pad=20)
    ax.set_ylabel('Number of Questions', fontsize=20, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', linewidth=0.8)
    
    # 移除第一个子图的x轴标签，使它们不重叠
    if row_idx == 0:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    else:
        ax.tick_params(axis='x', labelsize=20, rotation=90)
        ax.set_xlabel('Language Code', fontsize=20, fontweight='bold', labelpad=8)

# --- KEY CHANGE: 使用 fig.subplots_adjust() 手动控制布局 ---
# bottom: 将子图区域的底部边界向上移动到整个画布高度的 20% 处，为 X轴标签和图例留出空间。
# top: 稍微向下移动顶部边界，给标题留出空间。
# hspace: 也可以在这里设置，覆盖 GridSpec 的设置
fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.225, hspace=0.3)

# 设置图例
handles, labels = axes[0].get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
fig.legend(
    unique_labels.values(), unique_labels.keys(),
    loc='lower center',
    bbox_to_anchor=(0.53, 0.04), # 微调图例在画布上的位置
    ncol=5,
    fontsize=20,
    title='Thought Process Category',
    title_fontsize=20,
    frameon=False
)
# --- MODIFICATION END ---

plt.savefig("mtruth_llama_changes_shorter3.pdf", format='pdf', bbox_inches='tight')



