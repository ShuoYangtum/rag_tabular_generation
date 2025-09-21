# RAG+LLM表格生成工具

基于RAG（检索增强生成）和LLM（大语言模型）技术的表格数据预测工具。该工具能够利用训练集中的历史数据，通过RAG检索最相似的样本，结合LLM生成能力，对测试集的目标特征进行预测。

## 功能特点

- **RAG检索**: 从训练集中检索最相似的k个样本作为参考
- **LLM生成**: 使用指定的大语言模型进行目标特征预测
- **智能提示词**: 支持从文件加载自定义提示词模板，包含\boxed{}格式输出
- **缺失值处理**: 自动处理训练集和测试集中的缺失值
- **灵活配置**: 支持多种模型、相似度计算方法和生成参数
- **易于使用**: 简单的命令行接口和配置文件

## 项目结构

```
rag_tg/
├── model/
│   └── generation.py      # 主要生成算法
├── rag/
│   └── rag.py            # RAG检索算法
├── prompt/
│   ├── prompt.py         # 提示词模板类
│   └── prompt.txt        # 提示词模板文件
├── data/
│   ├── train.csv         # 示例训练集
│   └── test.csv          # 示例测试集
├── config.yaml           # 配置文件
├── run_sample.py         # 主入口文件
├── example_usage.py     # 使用示例
├── requirements.txt      # 依赖文件
└── README.md            # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据

准备两个CSV文件：
- **训练集**: 包含目标特征的完整数据
- **测试集**: 不包含目标特征，需要预测的数据

### 2. 配置参数

编辑 `config.yaml` 文件：

```yaml
# 模型配置
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"  # 可改为 Qwen3/Qwen3-1.7B
  device: "auto"

# RAG配置
rag:
  k: 5  # 检索最相似的5个样本
  similarity_method: "cosine"

# 数据路径
data:
  train_csv: "data/train.csv"
  test_csv: "data/test.csv"
  output_csv: "data/predictions.csv"

# 目标特征
target_feature:
  name: "target"  # 目标特征的列名
```

### 3. 运行预测

```bash
python run_sample.py
```

或者使用命令行参数：

```bash
python run_sample.py --train data/train.csv --test data/test.csv --output results.csv --target target_column
```

## 配置说明

### 模型配置
- `name`: 模型名称，支持Hugging Face上的模型
- `tokenizer`: tokenizer名称，默认与模型相同
- `device`: 设备选择（auto/cpu/cuda）
- `max_length`: 最大生成长度

### RAG配置
- `k`: 检索的相似样本数量
- `similarity_method`: 相似度计算方法（cosine/euclidean/manhattan）

### 生成配置
- `temperature`: 生成温度（0.0-1.0）
- `top_p`: nucleus sampling参数
- `do_sample`: 是否使用采样
- `max_new_tokens`: 最大新生成token数

## 支持的模型

- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2.5-14B-Instruct
- 其他Hugging Face上的文本生成模型

## 示例

### 数据格式

**训练集 (train.csv)**:
```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,high
2.1,4.3,6.5,medium
3.0,5.2,7.4,low
```

**测试集 (test.csv)**:
```csv
feature1,feature2,feature3
1.5,3.6,5.8
2.3,4.5,6.7
```

**输出 (predictions.csv)**:
```csv
feature1,feature2,feature3,target
1.5,3.6,5.8,high
2.3,4.5,6.7,medium
```

### 命令行使用

```bash
# 使用默认配置
python run_sample.py

# 指定模型和参数
python run_sample.py --model Qwen/Qwen2.5-7B-Instruct --k 10

# 指定数据文件
python run_sample.py --train my_train.csv --test my_test.csv --output my_results.csv --target my_target
```

## 工作原理

1. **数据预处理**: 加载训练集和测试集，处理缺失值
2. **RAG训练**: 使用训练集训练RAG检索器，建立特征相似度索引
3. **样本检索**: 对测试集的每个样本，检索最相似的k个训练样本
4. **提示词构建**: 将检索到的样本和当前样本构建成提示词
5. **LLM生成**: 使用大语言模型生成目标特征的预测值
6. **结果提取**: 从生成文本中提取\boxed{}中的预测值
7. **结果输出**: 将预测结果保存到CSV文件

## 提示词模板

项目支持自定义提示词模板，默认模板位于`prompt/prompt.txt`。模板包含以下占位符：

- `{num_examples}`: 检索到的相似样本数量
- `{examples}`: 相似样本的详细信息
- `{query_features}`: 当前待预测样本的特征

### 输出格式

提示词要求模型将预测结果包含在`\boxed{}`中，例如：
- `\boxed{high}` (分类特征)
- `\boxed{85.5}` (数值特征)

这样可以确保从生成文本中准确提取预测值。

## 注意事项

1. **模型下载**: 首次使用需要下载模型，请确保网络连接正常
2. **内存要求**: 大模型需要较多内存，建议使用GPU加速
3. **数据质量**: 训练集质量直接影响预测效果
4. **特征编码**: 分类特征会自动进行标签编码
5. **缺失值**: 数值型缺失值用均值填充，分类型用众数填充

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查网络连接
   - 确认模型名称正确
   - 尝试使用较小的模型

2. **内存不足**
   - 使用CPU模式
   - 选择较小的模型
   - 减少batch size

3. **预测结果为空**
   - 检查目标特征列名是否正确
   - 确认测试集格式正确
   - 查看错误日志

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。
