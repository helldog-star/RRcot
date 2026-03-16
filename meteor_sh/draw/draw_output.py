import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 数据 ────────────────────────────────
datasets = ['GSM8K', 'MMLU', 'GPQA', 'BBH']
methods = ['MemCoT', 'LightThinker']

data = {
    'MemCoT':       [3971, 4365, 3957, 3384],
    'LightThinker': [3206, 3402, 2979, 2784],
}

# ── 柱子样式 ─────────────────────────────
styles = {
    'MemCoT': {
        'color': '#CFCFCF',  # 灰色柱子
        'hatch': '/',
        'edgecolor': '#444444',
        'linewidth': 1.0
    },
    'LightThinker': {
        'color': '#5FAD93',  # 绿色柱子
        'hatch': '\\\\',
        'edgecolor': '#2E6F5E',
        'linewidth': 1.2
    }
}

# ── 布局 ───────────────────────────────
x = np.arange(len(datasets))
bar_width = 0.36

fig, ax = plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── 画柱子 ───────────────────────────────
for j, method in enumerate(methods):
    offset = (j - 0.5) * bar_width
    values = data[method]
    s = styles[method]

    bars = ax.bar(
        x + offset,
        values,
        width=bar_width,
        color=s['color'],
        hatch=s['hatch'],
        edgecolor=s['edgecolor'],
        linewidth=s['linewidth'],
        label=method
    )

    # 顶部数字标签
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            val + 35,
            str(val),
            ha='center',
            va='bottom',
            fontsize=9
        )

# ── x轴 ───────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)

# ── y轴 ───────────────────────────────
ax.set_ylabel('Generated Tokens ↓', fontsize=11)
ax.set_ylim(0,5100)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v,_: f'{int(v/1000)}k' if v>=1000 else str(int(v)))
)
ax.tick_params(axis='both', labelsize=10, direction='out', length=3)

# ── 边框 ───────────────────────────────
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('#555555')
    spine.set_linewidth(0.8)

# ── legend（只显示 hatch，不带填充） ───────
legend_patches = [
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='/', label='MemCoT'),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='\\\\', label='LightThinker')
]

ax.legend(handles=legend_patches, fontsize=10, framealpha=1, loc='upper right')

# ── 标题 ───────────────────────────────
ax.set_title(
    '(a) Average number of generated tokens across datasets',
    fontsize=11,
    pad=12
)

plt.tight_layout()

plt.savefig('generated_tokens_bar.pdf', dpi=300, bbox_inches='tight')
plt.savefig('generated_tokens_bar.png', dpi=300, bbox_inches='tight')
print("Saved: generated_tokens_bar.pdf & generated_tokens_bar.png")

plt.show()