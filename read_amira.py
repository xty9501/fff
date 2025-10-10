# import read_amira

# data = read_amira.read_amira("/project/xli281_uksr/mxia/data/2d_fluid_512_512_1001/6666.am")
import re
import base64
import numpy as np
from typing import Tuple, Dict, Optional

# ---------- 解析 Header ----------

def _read_header_binary(filepath: str) -> Tuple[bytes, str, int, Optional[bytes]]:
    """
    以二进制逐行读取，直至 '# Data section follows'
    返回：
      - header_bytes: 头部的原始字节（用于后续必要时参考）
      - header_str:   头部的文本（utf-8 解码，错误忽略）
      - data_pos:     数据起始的文件偏移（紧接 '# Data section follows' 之后，且跳过可选的'@标签'行）
      - data_tag:     若存在，形如 b'@1' 的数据标签行（不含换行），否则为 None
    """
    header_lines_bin = []
    data_pos = None
    data_tag = None

    with open(filepath, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("未找到 '# Data section follows' 标记。")
            header_lines_bin.append(line)
            if line.strip().lower() == b'# data section follows':
                # 到此为止：下一行可能是空行或 '@1' 之类的标签；数据从标签下一行开始
                # 记录当前位置（在这一行末尾）
                break

        # 可能跟一个空行
        pos_after_marker = f.tell()
        next_line = f.readline()
        # 如果是空白行，继续看下一行
        while next_line in (b'\n', b'\r\n', b'\r'):
            pos_after_marker = f.tell()
            next_line = f.readline()

        # 如果是 '@数字' 形式的 tag，则记录并把数据起点设为其下一行
        if next_line.strip().startswith(b'@'):
            data_tag = next_line.strip()
            data_pos = f.tell()
        else:
            # 否则数据就从刚读到的这一行开始，需要回退一行
            data_pos = pos_after_marker
            f.seek(data_pos, 0)

    header_bytes = b''.join(header_lines_bin)
    header_str = header_bytes.decode('utf-8', errors='ignore')
    return header_bytes, header_str, data_pos, data_tag


def _parse_metadata(header_str: str) -> Dict:
    """
    从 header 字符串中提取元数据：
      - shape: (nx, ny, nz)
      - base_type + components
      - dtype（含端序）
      - byteorder: '<' 或 '>'
      - encoding_hint: 'binary' | 'ascii' | 'base64'（来自 header 的显式/隐式判断）
    """
    # 维度
    shape_m = re.search(r"define\s+Lattice\s+(\d+)\s+(\d+)\s+(\d+)", header_str, flags=re.I)
    if not shape_m:
        raise ValueError("Header 中未找到 Lattice 维度定义。")
    nx, ny, nz = map(int, shape_m.groups())

    # 数据类型 & 分量数
    # 例：Lattice { float[2] Data } @1
    dtype_m = re.search(r"Lattice\s*\{\s*([A-Za-z]+)\s*(?:\[(\d+)\])?\s+Data\s*\}", header_str, flags=re.I)
    if not dtype_m:
        raise ValueError("Header 中未找到 Lattice 数据类型定义。")
    base_type = dtype_m.group(1).lower()
    components = int(dtype_m.group(2)) if dtype_m.group(2) else 1

    # 端序/编码判定
    first_line_m = re.search(r"#\s*AmiraMesh\s+([A-Z\-]+)", header_str, flags=re.I)
    enc_token = (first_line_m.group(1).upper() if first_line_m else "")
    if "ASCII" in enc_token:
        encoding_hint = "ascii"
        byteorder = "<"  # 对 ASCII 无意义，占位
    else:
        # 可能是 BINARY-LITTLE-ENDIAN / BINARY-BIG-ENDIAN
        if "LITTLE-ENDIAN" in enc_token:
            byteorder = "<"
        elif "BIG-ENDIAN" in enc_token:
            byteorder = ">"
        else:
            # 未写明，默认小端
            byteorder = "<"
        encoding_hint = "binary"

    # Base64 标记（若出现则覆盖）
    if re.search(r'Encoding\s*"Base64"', header_str, flags=re.I):
        encoding_hint = "base64"

    # dtype 映射
    _map = {
        "byte":   np.uint8,
        "char":   np.int8,
        "short":  np.int16,
        "ushort": np.uint16,
        "int":    np.int32,
        "uint":   np.uint32,
        "float":  np.float32,
        "double": np.float64,
    }
    if base_type not in _map:
        raise ValueError(f"未知的基础数据类型: {base_type}")

    base = np.dtype(_map[base_type])
    # 为其加上端序（ASCII 时无意义，但保持一致）
    dtype = np.dtype(byteorder + base.str[1:])

    return {
        "shape": (nx, ny, nz),
        "base_type": base_type,
        "components": components,
        "dtype": dtype,
        "byteorder": byteorder,
        "encoding_hint": encoding_hint,
    }

# ---------- 数据读取实现 ----------

def _read_binary(filepath: str, offset: int, dtype: np.dtype, count: int) -> np.ndarray:
    with open(filepath, 'rb') as f:
        f.seek(offset, 0)
        arr = np.fromfile(f, dtype=dtype, count=count)
    if arr.size != count:
        raise ValueError(f"二进制读取数量不匹配：期望 {count}，实际 {arr.size}")
    return arr


def _read_ascii(filepath: str, offset: int, dtype: np.dtype, count: int) -> np.ndarray:
    with open(filepath, 'rb') as f:
        f.seek(offset, 0)
        text = f.read().decode('utf-8', errors='ignore')
    arr = np.fromstring(text, sep=None, dtype=dtype, count=count)
    if arr.size != count:
        raise ValueError(f"ASCII 解析数量不匹配：期望 {count}，实际 {arr.size}")
    return arr


def _looks_like_base64(b: bytes) -> bool:
    # 取前面若干可见字符，判断是否只包含 base64 合法字符
    sample = b[:4096].strip()
    if not sample:
        return False
    # 去掉换行与空白
    sample = b"".join(sample.split())
    # 合法字符集合
    import string
    valid = set((string.ascii_letters + string.digits + "+/=").encode())
    return all(c in valid for c in sample)


def _read_base64(filepath: str, offset: int, dtype: np.dtype, count: int) -> np.ndarray:
    with open(filepath, 'rb') as f:
        f.seek(offset, 0)
        raw = f.read()
    # 去除空白并解码
    b64 = b"".join(raw.split())
    try:
        decoded = base64.b64decode(b64, validate=False)
    except Exception as e:
        raise ValueError(f"Base64 解码失败：{e}")
    arr = np.frombuffer(decoded, dtype=dtype, count=count)
    if arr.size != count:
        raise ValueError(f"Base64 解码数量不匹配：期望 {count}，实际 {arr.size}")
    return arr

# ---------- 顶层接口 ----------

def read_amira(filepath: str, save_npy: Optional[str] = None):
    """
    读取 Amira 文件，自动识别编码并解析为数组。
    返回：
      data: np.ndarray  (形状： (nz, ny, nx[, components]))
      meta: dict
    可选：
      save_npy: 保存为 .npy 文件路径
    """
    _, header_str, data_pos, data_tag = _read_header_binary(filepath)
    meta = _parse_metadata(header_str)

    nx, ny, nz = meta["shape"]
    comps = meta["components"]
    total = nx * ny * nz * comps

    # 选择读取策略
    encoding = meta["encoding_hint"]

    # 若标注为 binary，但数据区看起来像 base64，进行兜底尝试
    if encoding == "binary":
        with open(filepath, 'rb') as f:
            f.seek(data_pos, 0)
            probe = f.read(8192)
        if _looks_like_base64(probe):
            # 很多历史/特殊写法会把 Encoding 写在别处，这里兜底
            encoding = "base64"

    if encoding == "binary":
        flat = _read_binary(filepath, data_pos, meta["dtype"], total)
    elif encoding == "ascii":
        flat = _read_ascii(filepath, data_pos, meta["dtype"], total)
    elif encoding == "base64":
        flat = _read_base64(filepath, data_pos, meta["dtype"], total)
    else:
        raise ValueError(f"无法识别的数据编码：{encoding}")

    # 重塑
    if comps > 1:
        data = flat.reshape((nz, ny, nx, comps))
    else:
        data = flat.reshape((nz, ny, nx))

    # 可选保存
    if save_npy:
        np.save(save_npy, data)

    # 附加一些信息
    meta_out = dict(meta)
    meta_out["data_tag"] = data_tag.decode() if data_tag else None
    meta_out["array_shape"] = tuple(data.shape)

    return data, meta_out

data, meta = read_amira("/project/xli281_uksr/mxia/data/2d_fluid_512_512_1001/6666.am")
# header = read_amira_header("/project/xli281_uksr/mxia/data/2d_fluid_512_512_1001/6666.am")
# print_header_info(header)
print(meta)
print(data.shape)
u = np.ascontiguousarray(data[..., 0])  # (Z,Y,X)
v = np.ascontiguousarray(data[..., 1])
u.astype('<f4', copy=False).tofile('/project/xli281_uksr/mxia/data/2d_fluid_512_512_1001/u.bin')
v.astype('<f4', copy=False).tofile('/project/xli281_uksr/mxia/data/2d_fluid_512_512_1001/v.bin')