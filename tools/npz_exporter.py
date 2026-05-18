import numpy as np
import sys
import os

# 强制 numpy 打印所有数据时不使用省略号
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def extract_npz_to_txt_with_index(npz_path, txt_path):
    """
    读取 npz 文件，将内部所有数组无截断地导出到 txt，并为每行数据添加索引。
    """
    if not os.path.exists(npz_path):
        print(f"错误: 找不到文件 {npz_path}")
        return

    print(f"正在解析 {npz_path} ...")
    
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            with open(txt_path, 'w', encoding='utf-8') as f:
                
                for key in data.files:
                    arr = data[key]
                    
                    # 写入变量的元数据头部
                    f.write(f"{'='*60}\n")
                    f.write(f"变量名称 (Key): {key}\n")
                    f.write(f"原始形状 (Shape): {arr.shape}\n")
                    f.write(f"数据类型 (Dtype): {arr.dtype}\n")
                    f.write(f"{'-'*60}\n")
                    
                    # 1. 处理 0维标量
                    if arr.ndim == 0:
                        f.write(f"[0] {arr}\n")
                        
                    else:
                        # 2. 统一转换为 2D 矩阵以便添加行索引
                        if arr.ndim == 1:
                            # 1维数组转为 (N, 1)
                            reshaped_arr = arr.reshape(-1, 1)
                        elif arr.ndim == 2:
                            reshaped_arr = arr
                        else:
                            # 3维及以上高维数组：保留第0维，后面的维度全部展平
                            f.write(f"[注] 该变量为 {arr.ndim} 维数组，已保留第0维(行)，其余维度展平。\n")
                            reshaped_arr = arr.reshape(arr.shape[0], -1)
                        
                        rows, cols = reshaped_arr.shape
                        
                        if cols == 0:
                            f.write("[空数组]\n\n")
                            continue
                            
                        # 生成行索引 (0 到 rows-1)
                        indices = np.arange(rows)
                        
                        # 将索引列拼接到数据矩阵的最左侧
                        # 注意：如果原数据是 float，此处 indices 会被暂时向上转型为 float
                        stacked_arr = np.column_stack((indices, reshaped_arr))
                        
                        # 3. 动态生成格式化字符串 (核心技巧)
                        # 第一列(索引)强制转回整数 [%d]，后面的数据列使用 %s 保持最高精度
                        data_fmts = ', '.join(['%s'] * cols)
                        row_fmt = f"[%d]\t{data_fmts}"  # 用制表符 \t 隔开索引和数据，更美观
                        
                        # 写入文件
                        np.savetxt(f, stacked_arr, fmt=row_fmt)
                    
                    f.write("\n\n") # 不同变量之间留白
                    print(f"  - 成功导出变量: {key} (Shape: {arr.shape})")
                    
        print(f"\n全部导出完成！带有索引的数据已保存至: {txt_path}")
        
    except Exception as e:
        print(f"导出过程中出现错误: {e}")

if __name__ == "__main__":
    # ----------------------------------------
    # 在这里修改你的文件路径
    # ----------------------------------------
    INPUT_NPZ_FILE = "dumps/2/nan_002_20260518-195759.npz"   # 替换为你的 npz 文件路径
    OUTPUT_TXT_FILE = "dumps/2/nan_002_20260518-195759.txt"     # 替换为你想要保存的 txt 文件路径
    
    extract_npz_to_txt_with_index(INPUT_NPZ_FILE, OUTPUT_TXT_FILE)