"""
提示词模板模块
用于构建RAG+LLM生成所需的提示词
"""

import os
from typing import List, Dict, Any
import pandas as pd


class PromptTemplate:
    """提示词模板类"""
    
    def __init__(self, template_file: str = None):
        """
        初始化提示词模板
        
        Args:
            template_file: 模板文件路径，如果为None则使用默认模板
        """
        if template_file and os.path.exists(template_file):
            self.template = self._load_template_from_file(template_file)
        else:
            self.template = self._get_default_template()
    
    def _load_template_from_file(self, template_file: str) -> str:
        """
        从文件加载提示词模板
        
        Args:
            template_file: 模板文件路径
            
        Returns:
            模板字符串
        """
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"加载模板文件失败: {e}")
            return self._get_default_template()
    
    def _get_default_template(self) -> str:
        """
        获取默认的提示词模板
        
        Returns:
            提示词模板字符串
        """
        template = """你是一个专业的数据分析师，擅长根据历史数据预测目标特征的值。

## 任务描述
你需要根据给定的特征值，预测目标特征的值。我会提供一些相似的历史样本作为参考。

## 历史参考样本
以下是训练集中最相似的{num_examples}个样本：

{examples}

## 当前待预测样本
请根据以下特征值预测目标特征的值：

{query_features}

## 预测要求
1. 仔细分析历史样本中特征值与目标特征的关系
2. 考虑当前样本的特征值与历史样本的相似性
3. 基于历史模式给出合理的预测
4. 如果目标特征是数值型，请给出具体的数值
5. 如果目标特征是分类型，请给出具体的类别

## 输出格式
请直接生成目标特征的值，并将预测结果包含在\\boxed{{}}中。

例如：\\boxed{{high}} 或 \\boxed{{85.5}}

预测结果："""
        return template
    
    def format_prompt(self, 
                     query_features: Dict[str, Any], 
                     examples: List[Dict[str, Any]], 
                     target_feature: str) -> str:
        """
        格式化提示词
        
        Args:
            query_features: 查询样本的特征字典
            examples: RAG检索到的相似样本列表
            target_feature: 目标特征名称
            
        Returns:
            格式化后的提示词
        """
        # 格式化查询特征
        query_text = self._format_features(query_features)
        
        # 格式化示例
        examples_text = self._format_examples(examples, target_feature)
        
        # 替换模板中的占位符
        prompt = self.template.format(
            num_examples=len(examples),
            examples=examples_text,
            query_features=query_text,
            target_feature=target_feature
        )
        
        return prompt
    
    def _format_features(self, features: Dict[str, Any]) -> str:
        """
        格式化特征字典为文本
        
        Args:
            features: 特征字典
            
        Returns:
            格式化后的特征文本
        """
        feature_lines = []
        for key, value in features.items():
            if pd.isna(value):
                feature_lines.append(f"- {key}: 缺失值")
            else:
                feature_lines.append(f"- {key}: {value}")
        
        return "\n".join(feature_lines)
    
    def _format_examples(self, examples: List[Dict[str, Any]], target_feature: str) -> str:
        """
        格式化示例为文本
        
        Args:
            examples: 示例列表
            target_feature: 目标特征名称
            
        Returns:
            格式化后的示例文本
        """
        example_texts = []
        
        for i, example in enumerate(examples, 1):
            features = example['features']
            target_value = example['target']
            similarity = example['similarity']
            
            feature_lines = []
            for key, value in features.items():
                if pd.isna(value):
                    feature_lines.append(f"    {key}: 缺失值")
                else:
                    feature_lines.append(f"    {key}: {value}")
            
            example_text = f"""样本 {i} (相似度: {similarity:.3f}):
{chr(10).join(feature_lines)}
    {target_feature}: {target_value}"""
            
            example_texts.append(example_text)
        
        return "\n\n".join(example_texts)
    
    def set_custom_template(self, template: str):
        """
        设置自定义提示词模板
        
        Args:
            template: 自定义模板字符串，应包含以下占位符：
                     - {num_examples}: 示例数量
                     - {examples}: 示例文本
                     - {query_features}: 查询特征文本
        """
        self.template = template
    
    def get_template(self) -> str:
        """
        获取当前模板
        
        Returns:
            当前模板字符串
        """
        return self.template
