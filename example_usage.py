"""
RAG+LLM表格生成工具使用示例
演示如何使用生成器进行预测
"""

import pandas as pd
from model.generation import RAGLLMGenerator


def main():
    """示例主函数"""
    print("RAG+LLM表格生成工具使用示例")
    print("=" * 50)
    
    # 创建生成器
    generator = RAGLLMGenerator("config.yaml")
    
    # 训练数据路径
    train_csv = "data/train.csv"
    test_csv = "data/test.csv"
    output_csv = "data/predictions.csv"
    target_feature = "target"
    
    try:
        print("1. 加载训练数据...")
        train_df = pd.read_csv(train_csv)
        print(f"训练数据形状: {train_df.shape}")
        print("训练数据预览:")
        print(train_df.head())
        
        print("\n2. 加载测试数据...")
        test_df = pd.read_csv(test_csv)
        print(f"测试数据形状: {test_df.shape}")
        print("测试数据预览:")
        print(test_df.head())
        
        print("\n3. 开始训练RAG检索器...")
        generator.train(train_csv, target_feature)
        
        print("\n4. 开始预测...")
        result_df = generator.predict(test_csv, output_csv, target_feature)
        
        print("\n5. 预测结果:")
        print(result_df)
        
        print(f"\n预测完成! 结果已保存到: {output_csv}")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
