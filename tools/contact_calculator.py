import numpy as np
import pandas as pd

# 1. 读取你上传的 CSV 文件
df = pd.read_csv('tools/contactpair.csv', index_col=0)

link_names = df.columns.tolist()
n = len(link_names)

# 2. 提取并清理数据（处理表格中的空白和逗号）
contact_matrix = np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(n):
        val = df.values[i, j]
        if pd.notna(val) and str(val).strip() != '':
            try:
                contact_matrix[i, j] = int(float(val))
            except ValueError:
                pass

# 3. 将你的下三角矩阵补全为对称矩阵
for i in range(n):
    for j in range(i, n):
        # 只要有一方标记了 1，就视为两者需要碰撞
        contact_matrix[i, j] = max(contact_matrix[i, j], contact_matrix[j, i])
        contact_matrix[j, i] = contact_matrix[i, j]

# 取消物体自身的碰撞（通常不需要自碰）
for i in range(n):
    contact_matrix[i, i] = 0

# 4. 运行等价类合并算法
unique_profiles, inverse_indices = np.unique(contact_matrix, axis=0, return_inverse=True)
num_groups = unique_profiles.shape[0]

if num_groups > 32:
    print(f"警告：独立的碰撞组数量 ({num_groups}) 超过了 32 位的限制！请考虑合并物体或改用 <contact> 标签。")
else:
    group_contype = np.zeros(num_groups, dtype=int)
    group_conaffinity = np.zeros(num_groups, dtype=int)
    
    for i in range(num_groups):
        group_contype[i] = 1 << i  # 独热码
        
    for i in range(num_groups):
        for j in range(num_groups):
            rep_idx = np.where(inverse_indices == j)[0][0]
            if unique_profiles[i, rep_idx] == 1:
                group_conaffinity[i] |= group_contype[j]
                
    contypes = group_contype[inverse_indices]
    conaffinities = group_conaffinity[inverse_indices]
    
    # 5. 直接输出结果
    print(f"{'物体名称 (Link)'.ljust(20)}| {'Contype'.ljust(10)} | {'Conaffinity'.ljust(10)}")
    print("-" * 50)
    for i in range(n):
        print(f'{link_names[i].ljust(20)} contype="{contypes[i]}" conaffinity="{conaffinities[i]}"')