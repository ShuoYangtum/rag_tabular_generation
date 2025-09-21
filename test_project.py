"""
项目完整性测试脚本
验证所有模块是否能正常工作
"""

import os
import sys
import pandas as pd
import yaml

def test_imports():
    """测试所有模块的导入"""
    print("1. 测试模块导入...")
    
    try:
        from rag.rag import RAGRetriever
        print("   ✓ RAG模块导入成功")
    except Exception as e:
        print(f"   ✗ RAG模块导入失败: {e}")
        return False
    
    try:
        from prompt.prompt import PromptTemplate
        print("   ✓ 提示词模块导入成功")
    except Exception as e:
        print(f"   ✗ 提示词模块导入失败: {e}")
        return False
    
    try:
        from model.generation import RAGLLMGenerator
        print("   ✓ 生成模块导入成功")
    except Exception as e:
        print(f"   ✗ 生成模块导入失败: {e}")
        return False
    
    return True

def test_config():
    """测试配置文件"""
    print("\n2. 测试配置文件...")
    
    if not os.path.exists("config.yaml"):
        print("   ✗ 配置文件不存在")
        return False
    
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("   ✓ 配置文件格式正确")
        
        # 检查必要的配置项
        required_keys = ['model', 'rag', 'data', 'target_feature', 'generation']
        for key in required_keys:
            if key not in config:
                print(f"   ✗ 缺少配置项: {key}")
                return False
        
        print("   ✓ 配置文件内容完整")
        return True
        
    except Exception as e:
        print(f"   ✗ 配置文件解析失败: {e}")
        return False

def test_data_files():
    """测试数据文件"""
    print("\n3. 测试数据文件...")
    
    # 检查示例数据文件
    data_files = ["data/train.csv", "data/test.csv"]
    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"   ✗ 数据文件不存在: {file_path}")
            return False
        
        try:
            df = pd.read_csv(file_path)
            print(f"   ✓ {file_path} 读取成功 (形状: {df.shape})")
        except Exception as e:
            print(f"   ✗ {file_path} 读取失败: {e}")
            return False
    
    return True

def test_prompt_template():
    """测试提示词模板"""
    print("\n4. 测试提示词模板...")
    
    # 检查prompt.txt文件
    if not os.path.exists("prompt/prompt.txt"):
        print("   ✗ 提示词模板文件不存在")
        return False
    
    try:
        from prompt.prompt import PromptTemplate
        template = PromptTemplate("prompt/prompt.txt")
        
        # 测试模板格式化
        test_features = {"age": 25, "income": 50000}
        test_examples = [{"features": {"age": 26, "income": 52000}, "target": "high", "similarity": 0.95}]
        
        prompt = template.format_prompt(test_features, test_examples, "target")
        
        if "{num_examples}" in prompt or "{examples}" in prompt or "{query_features}" in prompt:
            print("   ✗ 提示词模板格式化失败，仍有未替换的占位符")
            return False
        
        print("   ✓ 提示词模板工作正常")
        return True
        
    except Exception as e:
        print(f"   ✗ 提示词模板测试失败: {e}")
        return False

def test_rag_functionality():
    """测试RAG功能"""
    print("\n5. 测试RAG功能...")
    
    try:
        from rag.rag import RAGRetriever
        
        # 创建测试数据
        train_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': ['A', 'B', 'A', 'B', 'A']
        })
        
        # 测试RAG检索器
        rag = RAGRetriever(k=3)
        rag.fit(train_data, 'target')
        
        # 测试检索
        test_sample = pd.Series({'feature1': 2.5, 'feature2': 25})
        results = rag.retrieve(test_sample)
        
        if len(results) != 3:
            print(f"   ✗ RAG检索结果数量不正确: {len(results)}")
            return False
        
        print("   ✓ RAG功能工作正常")
        return True
        
    except Exception as e:
        print(f"   ✗ RAG功能测试失败: {e}")
        return False

def test_prediction_extraction():
    """测试预测值提取"""
    print("\n6. 测试预测值提取...")
    
    try:
        from model.generation import RAGLLMGenerator
        
        generator = RAGLLMGenerator("config.yaml")
        
        # 测试不同的输出格式
        test_cases = [
            ("\\boxed{high}", "high"),
            ("\\boxed{85.5}", "85.5"),
            ("预测结果：high", "high"),
            ("high", "high"),
            ("答案是：\\boxed{medium}", "medium")
        ]
        
        for input_text, expected in test_cases:
            result = generator._clean_prediction(input_text)
            if result != expected:
                print(f"   ✗ 预测值提取失败: '{input_text}' -> '{result}' (期望: '{expected}')")
                return False
        
        print("   ✓ 预测值提取功能正常")
        return True
        
    except Exception as e:
        print(f"   ✗ 预测值提取测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("RAG+LLM表格生成工具 - 项目完整性测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_data_files,
        test_prompt_template,
        test_rag_functionality,
        test_prediction_extraction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过！项目结构完整。")
        return True
    else:
        print("✗ 部分测试失败，请检查相关模块。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
