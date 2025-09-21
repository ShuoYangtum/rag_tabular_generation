"""
RAG+LLM表格生成主入口文件
执行生成时通过 python run_sample.py 来启动
"""

import argparse
import yaml
import os
import sys
from model.generation import RAGLLMGenerator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RAG+LLM表格生成工具')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--train', type=str, 
                       help='训练集CSV文件路径 (覆盖配置文件中的设置)')
    parser.add_argument('--test', type=str, 
                       help='测试集CSV文件路径 (覆盖配置文件中的设置)')
    parser.add_argument('--output', type=str, 
                       help='输出CSV文件路径 (覆盖配置文件中的设置)')
    parser.add_argument('--target', type=str, 
                       help='目标特征列名 (覆盖配置文件中的设置)')
    parser.add_argument('--model', type=str, 
                       help='模型名称 (覆盖配置文件中的设置)')
    parser.add_argument('--k', type=int, 
                       help='RAG检索的样本数量 (覆盖配置文件中的设置)')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 '{args.config}' 不存在")
        print("请确保配置文件存在，或使用 --config 参数指定正确的配置文件路径")
        sys.exit(1)
    
    try:
        # 创建生成器
        generator = RAGLLMGenerator(args.config)
        
        # 从配置文件获取参数
        config = generator.config
        
        # 使用命令行参数覆盖配置文件设置
        train_csv = args.train or config['data']['train_csv']
        test_csv = args.test or config['data']['test_csv']
        output_csv = args.output or config['data']['output_csv']
        target_feature = args.target or config['target_feature']['name']
        
        # 检查文件是否存在
        if not os.path.exists(train_csv):
            print(f"错误: 训练集文件 '{train_csv}' 不存在")
            sys.exit(1)
        
        if not os.path.exists(test_csv):
            print(f"错误: 测试集文件 '{test_csv}' 不存在")
            sys.exit(1)
        
        # 更新配置（如果提供了命令行参数）
        if args.model:
            config['model']['name'] = args.model
        if args.k:
            config['rag']['k'] = args.k
        
        print("=" * 60)
        print("RAG+LLM表格生成工具")
        print("=" * 60)
        print(f"模型: {config['model']['name']}")
        print(f"训练集: {train_csv}")
        print(f"测试集: {test_csv}")
        print(f"输出文件: {output_csv}")
        print(f"目标特征: {target_feature}")
        print(f"RAG检索数量: {config['rag']['k']}")
        print("=" * 60)
        
        # 训练RAG检索器
        print("\n步骤1: 训练RAG检索器")
        generator.train(train_csv, target_feature)
        
        # 进行预测
        print("\n步骤2: 开始预测")
        result_df = generator.predict(test_csv, output_csv, target_feature)
        
        print("\n" + "=" * 60)
        print("预测完成!")
        print(f"结果已保存到: {output_csv}")
        print("=" * 60)
        
        # 显示预测结果统计
        predictions = result_df[target_feature].dropna()
        print(f"\n预测结果统计:")
        print(f"- 总样本数: {len(result_df)}")
        print(f"- 成功预测: {len(predictions)}")
        print(f"- 预测失败: {len(result_df) - len(predictions)}")
        
        if len(predictions) > 0:
            print(f"\n预测结果预览:")
            print(result_df.head())
        
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def demo():
    """演示函数，展示如何使用生成器"""
    print("RAG+LLM表格生成工具演示")
    print("=" * 40)
    
    # 创建示例配置
    demo_config = {
        'model': {'name': 'Qwen/Qwen2.5-1.5B-Instruct', 'device': 'auto'},
        'rag': {'k': 3, 'similarity_method': 'cosine'},
        'data': {
            'train_csv': 'data/train.csv',
            'test_csv': 'data/test.csv', 
            'output_csv': 'data/predictions.csv'
        },
        'target_feature': {'name': 'target'},
        'generation': {
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True,
            'max_new_tokens': 50
        }
    }
    
    print("示例配置:")
    print(yaml.dump(demo_config, default_flow_style=False, allow_unicode=True))
    
    print("\n使用方法:")
    print("1. 准备训练集和测试集CSV文件")
    print("2. 修改config.yaml配置文件")
    print("3. 运行: python run_sample.py")
    print("\n或者使用命令行参数:")
    print("python run_sample.py --train data/train.csv --test data/test.csv --output results.csv --target target_column")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 如果没有参数，显示帮助信息
        print("RAG+LLM表格生成工具")
        print("=" * 40)
        print("使用方法:")
        print("python run_sample.py                    # 使用默认配置")
        print("python run_sample.py --help             # 显示帮助信息")
        print("python run_sample.py --config my_config.yaml  # 使用自定义配置")
        print("\n示例:")
        print("python run_sample.py --train data/train.csv --test data/test.csv --output results.csv --target target_column")
        print("\n详细帮助:")
        main()
    else:
        main()
