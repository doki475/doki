# 论文查重系统

一个基于Python的论文查重系统，用于检测两篇论文之间的重复率。

## 功能

- 支持中英文文本的查重比较
- 使用多种算法组合提高检测准确性
- 支持文件输入输出
- 自动检测文件编码（UTF-8/UTF-16/GBK/GB2312）

## 算法说明

系统使用加权组合算法计算相似度：

1. **字符n-gram Jaccard相似度**（50%权重）
   - 2-gram和3-gram Jaccard系数的加权组合
   - 捕捉字符级别的精确匹配

2. **字符频率余弦相似度**（30%权重）
   - 基于字符频次向量的余弦相似度
   - 捕捉整体的字符分布相似性

3. **词语Jaccard相似度**（20%权重）
   - 使用jieba分词后进行词语级别的Jaccard比较
   - 捕捉语义级别的相似性

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python main.py <原文文件路径> <抄袭版文件路径> <答案文件路径>
```

### 示例

```bash
python main.py orig.txt plagiarized.txt answer.txt
```

输出文件内容为浮点数，精确到小数点后两位，表示重复率（0.00 ~ 1.00）。

## 测试

```bash
python -m pytest tests/ -v
# 或
python -m unittest discover tests/ -v
```

生成覆盖率报告：

```bash
pip install coverage
coverage run -m pytest tests/ -v
coverage report
coverage html  # 生成HTML报告
```

## 项目结构

```
├── main.py                  # 程序入口
├── checker/                 # 核心模块
│   ├── __init__.py
│   ├── algorithm.py         # 查重算法
│   └── preprocessor.py      # 文本预处理
├── tests/                   # 测试模块
│   ├── __init__.py
│   ├── test_core.py         # 单元测试（30+测试用例）
│   └── tests_data/          # 测试数据
├── requirements.txt         # 依赖
├── PSP.md                   # PSP时间记录
└── README.md                # 项目说明
```