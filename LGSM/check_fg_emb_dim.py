import pickle

# 加载官能团嵌入字典
with open('./fg2emb.pkl', 'rb') as f:
    fg2emb = pickle.load(f)

# 查看字典中的键数量
print(f"官能团种类数量: {len(fg2emb.keys())}")

# 查看第一个官能团的嵌入维度
if len(fg2emb) > 0:
    first_fg = list(fg2emb.keys())[0]
    first_emb = fg2emb[first_fg]
    print(f"第一个官能团 '{first_fg}' 的嵌入维度: {len(first_emb)}")
    print(f"嵌入向量示例: {first_emb[:5]}...")

# 查看几个随机的官能团嵌入
print("\n随机查看几个官能团的嵌入:")
for i, (fg_name, emb) in enumerate(list(fg2emb.items())[:5]):
    print(f"{i+1}. {fg_name}: 维度={len(emb)}")