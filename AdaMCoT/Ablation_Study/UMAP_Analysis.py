import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_parallel_data(data_dir, languages):
    """
    从多个并行文本文件中加载数据。

    Args:
        data_dir (str): 存放文本文件的目录。
        languages (dict): 一个字典，键是语言代码(e.g., 'en')，值是文件名(e.g., 'en.txt')。

    Returns:
        tuple: (all_sentences, all_labels)
               all_sentences是一个包含所有句子的大列表。
               all_labels是与句子一一对应的语言标签列表。
    """
    lang_data = {}
    for lang_code, filename in languages.items():
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lang_data[lang_code] = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"错误：文件 {file_path} 未找到。")
            return None, None

    # 检查所有语言的句子数量是否相同
    num_sentences = len(list(lang_data.values())[0])
    if not all(len(sents) == num_sentences for sents in lang_data.values()):
        print("错误：各语言文件中的句子数量不一致。")
        return None, None

    print(f"成功加载 {len(languages)} 种语言，每种语言 {num_sentences} 句话。")

    all_sentences = []
    all_labels = []

    # 使用 zip 将并行句子组合在一起
    # [(sent_en_1, sent_zh_1, ...), (sent_en_2, sent_zh_2, ...)]
    parallel_sentences = zip(*[lang_data[lang] for lang in languages.keys()])

    for sentence_group in parallel_sentences:
        for i, lang_code in enumerate(languages.keys()):
            all_sentences.append(sentence_group[i])
            all_labels.append(lang_code)

    return all_sentences, all_labels

# --- 使用示例 ---
# 假设你的文件在 'data' 文件夹下
DATA_DIRECTORY = '/home/weihua/Huangxin_Work/LogitLens/Flores-200/filter_lang_files' # <-- 修改为你的数据目录
LANGUAGES_FILES = {
    'ar': 'ar.concat',
    'bn': 'bn.concat',
    'ca': 'ca.concat',
    'da': 'da.concat',
    'de': 'de.concat',
    'en': 'en.concat',
    'es': 'es.concat',
    'eu': 'eu.concat',
    'fr': 'fr.concat',
    'gu': 'gu.concat',
    'hi': 'hi.concat',
    'hr': 'hr.concat',
    'hu': 'hu.concat',
    'hy': 'hy.concat',
    'id': 'id.concat',
    'it': 'it.concat',
    'kn': 'kn.concat',
    'ml': 'ml.concat',
    'mr': 'mr.concat',
    'ne': 'ne.concat',
    'nl': 'nl.concat',
    'pt': 'pt.concat',
    'ro': 'ro.concat',
    'ru': 'ru.concat',
    'sk': 'sk.concat',
    'sr': 'sr.concat',
    'sv': 'sv.concat',
    'ta': 'ta.concat',
    'te': 'te.concat',
    'uk': 'uk.concat',
    'vi': 'vi.concat',
    'zh': 'zh.concat',
}

sentences, labels = load_parallel_data(DATA_DIRECTORY, LANGUAGES_FILES)

# 打印一些示例数据来验证
if sentences:
    print("\n数据加载示例:")
    print(f"句子: {sentences[0]}, 语言: {labels[0]}")
    print(f"句子: {sentences[1]}, 语言: {labels[1]}")
    print(f"总句子数: {len(sentences)}")

# 1. 加载模型和分词器
# 确保你已通过 huggingface-cli login 登录
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_id = "./Model/Qwen-2.5-7B-Adop/"
# model_id = "Qwen/Qwen2.5-7B-Instruct"


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_averaged_hidden_states(sentences, model, tokenizer, layers_to_average):
    """
    输入句子列表，返回在指定层级上平均后的句子级表征。
    """
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    attention_mask = inputs['attention_mask']
    
    # 初始化一个tensor来累加指定层的向量
    # 形状: (batch_size, hidden_size)
    aggregated_embeddings = torch.zeros(
        (len(sentences), model.config.hidden_size), 
        device=model.device, 
        dtype=torch.float32
    )

    # 遍历需要平均的层
    for layer_idx in layers_to_average:
        layer_hidden_states = hidden_states[layer_idx]

        # 进行均值池化（得到当前层的句子向量）
        last_hidden_state_masked = layer_hidden_states * attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(last_hidden_state_masked, dim=1)
        num_valid_tokens = torch.sum(attention_mask, dim=1, keepdim=True)
        sentence_embedding = sum_embeddings / num_valid_tokens
        
        # 累加到总和中
        aggregated_embeddings += sentence_embedding

    # 对累加后的向量求平均
    final_averaged_embeddings = aggregated_embeddings / len(layers_to_average)

    return final_averaged_embeddings.cpu().numpy()

# --- 运行提取过程 ---
if sentences:
    # 定义要平均的层（18到31层）
    LAYERS_TO_AVERAGE = list(range(16, 28)) # range(18, 32) 包含 18, 19, ..., 31

    batch_size = 16 # 根据你的 VRAM 调整
    all_aggregated_embeddings = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size

    print(f"\n开始提取和平均层 {LAYERS_TO_AVERAGE} 的向量...")
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        
        # 调用核心函数
        batch_embeddings = get_averaged_hidden_states(batch_sentences, model, tokenizer, LAYERS_TO_AVERAGE)
        all_aggregated_embeddings.append(batch_embeddings)
        
        print(f"处理完成批次 {i//batch_size + 1}/{total_batches}")

    # 将所有批次的结果合并成一个大的Numpy数组
    final_embeddings = np.vstack(all_aggregated_embeddings)

    print("\n向量提取和平均完成！")
    print(f"最终的 embedding 形状: {final_embeddings.shape}")


def visualize_embeddings(embeddings, labels, title):
    """
    使用 UMAP 降维并可视化，使用高区分度的颜色。
    """
    print("\n正在进行UMAP降维...")
    # 1. 标准化数据
    scaled_embeddings = StandardScaler().fit_transform(embeddings)

    # 2. UMAP 降维
    reducer = umap.UMAP(n_neighbors=20, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
    reduced_embeddings = reducer.fit_transform(scaled_embeddings)
    print("降维完成。")

    # 3. 创建 DataFrame 用于绘图
    plot_df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'language': labels
    })

    # --- 颜色处理部分 (核心改动) ---
    # 获取所有独特的语言标签，并排序以保证颜色分配的一致性
    unique_languages = sorted(plot_df['language'].unique())
    num_languages = len(unique_languages)

    # 组合多个高质量的分类型调色盘来创建足够的颜色
    # tab20 有 20 种颜色, Set3 有 12 种颜色。20 + 12 = 32，正好够用！
    palette_tab20 = sns.color_palette("tab20", 20)
    palette_set3 = sns.color_palette("Set3", 12)
    
    # 如果语言数量超过我们准备的，就回退到 hls
    if num_languages > len(palette_tab20) + len(palette_set3):
        print(f"警告: 语言数量 ({num_languages}) 太多，回退到 'hls' 调色盘。")
        colors = sns.color_palette("hls", num_languages)
    else:
        # 组合调色盘
        colors = (palette_tab20 + palette_set3)[:num_languages]

    # 创建从语言到颜色的映射字典
    language_color_map = {lang: color for lang, color in zip(unique_languages, colors)}
    # --- 颜色处理结束 ---

    # 4. 绘图
    plt.figure(figsize=(16, 14)) # 稍微增大图像尺寸以容纳图例
    ax = sns.scatterplot(
        data=plot_df,
        x='x',
        y='y',
        hue='language',
        # 直接将颜色映射字典传递给 palette 参数
        palette=language_color_map,
        alpha=0.7,
        s=30
    )
    plt.title(title, fontsize=20, pad=20)
    plt.xlabel("UMAP Dimension 1", fontsize=14)
    plt.ylabel("UMAP Dimension 2", fontsize=14)
    
    # 优化图例，如果类别太多，图例会很大
    # 将图例放在图的外面，并调整列数
    ax.legend(title='Language', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., ncol=2, markerscale=1.5)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 使用 bbox_inches="tight" 来确保图例不会被裁切
    plt.savefig("Language_Distribution_High_Contrast_Qwen_Adop.pdf", format='pdf', dpi=600, bbox_inches="tight")
    # plt.savefig("Language_Distribution_High_Contrast.png", format='png', dpi=300, bbox_inches="tight")
    print("\n已保存高对比度图像 'Language_Distribution_High_Contrast.pdf' 和 '.png'")
    # plt.show()

# --- 运行可视化 ---
if 'final_embeddings' in locals():
    visualization_title = f'UMAP of Qwen 2.5 Adop Embeddings (Averaged Layers {LAYERS_TO_AVERAGE[0]}-{LAYERS_TO_AVERAGE[-1]})'
    visualize_embeddings(final_embeddings, labels, visualization_title)