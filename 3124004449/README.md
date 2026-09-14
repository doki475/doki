# 论文查重系统

Python 论文查重作业，支持命令行文件输入输出。

## 运行

```bash
python -m pip install -r requirements.txt
python main.py <原文绝对路径> <抄袭版绝对路径> <答案绝对路径>
```

## 测试

```bash
python -m unittest discover tests -v
```

核心代码位于 `main.py` 和 `checker/`，测试代码位于 `tests/`。
