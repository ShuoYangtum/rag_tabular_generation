"""
主要生成算法模块
整合RAG检索和LLM生成功能
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict, Any, Optional
import yaml
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from rag.rag import RAGRetriever
from prompt.prompt import PromptTemplate


class RAGLLMGenerator:
    """RAG+LLM生成器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化生成器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.rag_retriever = None
        # 尝试从prompt.txt文件加载模板，如果不存在则使用默认模板
        prompt_file = "prompt/prompt.txt"
        self.prompt_template = PromptTemplate(prompt_file)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_model(self):
        """设置模型和tokenizer"""
        model_name = self.config['model']['name']
        tokenizer_name = self.config['model'].get('tokenizer') or model_name
        device = self.config['model']['device']
        
        print(f"正在加载模型: {model_name}")
        print(f"正在加载tokenizer: {tokenizer_name}")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device,
                trust_remote_code=True
            )
            
            # 创建pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            print("模型加载完成!")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请检查模型名称是否正确，或者网络连接是否正常")
            raise
    
    def _setup_rag(self):
        """设置RAG检索器"""
        k = self.config['rag']['k']
        similarity_method = self.config['rag']['similarity_method']
        
        self.rag_retriever = RAGRetriever(k=k, similarity_method=similarity_method)
        print(f"RAG检索器初始化完成 (k={k}, method={similarity_method})")
    
    def train(self, train_csv_path: str, target_feature: str):
        """
        训练RAG检索器
        
        Args:
            train_csv_path: 训练集CSV文件路径
            target_feature: 目标特征列名
        """
        print(f"正在加载训练数据: {train_csv_path}")
        
        # 加载训练数据
        train_df = pd.read_csv(train_csv_path)
        print(f"训练数据形状: {train_df.shape}")
        print(f"训练数据列: {list(train_df.columns)}")
        
        # 检查目标特征是否存在
        if target_feature not in train_df.columns:
            raise ValueError(f"目标特征 '{target_feature}' 不在训练数据中")
        
        # 设置RAG检索器
        self._setup_rag()
        
        # 训练RAG检索器
        print("正在训练RAG检索器...")
        self.rag_retriever.fit(train_df, target_feature)
        
        # 设置模型
        self._setup_model()
        
        print("训练完成!")
    
    def predict(self, test_csv_path: str, output_csv_path: str, target_feature: str):
        """
        对测试集进行预测
        
        Args:
            test_csv_path: 测试集CSV文件路径
            output_csv_path: 输出CSV文件路径
            target_feature: 目标特征列名
        """
        print(f"正在加载测试数据: {test_csv_path}")
        
        # 加载测试数据
        test_df = pd.read_csv(test_csv_path)
        print(f"测试数据形状: {test_df.shape}")
        print(f"测试数据列: {list(test_df.columns)}")
        
        # 检查RAG检索器是否已训练
        if self.rag_retriever is None:
            raise ValueError("RAG检索器尚未训练，请先调用train方法")
        
        # 检查模型是否已加载
        if self.pipeline is None:
            raise ValueError("模型尚未加载，请先调用train方法")
        
        # 创建输出DataFrame
        output_df = test_df.copy()
        predictions = []
        
        print("开始预测...")
        
        # 对每个测试样本进行预测
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="预测进度"):
            try:
                # RAG检索相似样本
                similar_samples = self.rag_retriever.retrieve(row)
                
                # 构建提示词
                prompt = self.prompt_template.format_prompt(
                    query_features=row.to_dict(),
                    examples=similar_samples,
                    target_feature=target_feature
                )
                
                # LLM生成预测
                prediction = self._generate_prediction(prompt)
                
                predictions.append(prediction)
                
            except Exception as e:
                print(f"预测第{idx+1}行时出错: {e}")
                predictions.append(None)
        
        # 添加预测结果到输出DataFrame
        output_df[target_feature] = predictions
        
        # 保存结果
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        output_df.to_csv(output_csv_path, index=False)
        
        print(f"预测完成! 结果已保存到: {output_csv_path}")
        
        # 统计预测结果
        valid_predictions = [p for p in predictions if p is not None]
        print(f"成功预测: {len(valid_predictions)}/{len(predictions)} 个样本")
        
        return output_df
    
    def _generate_prediction(self, prompt: str) -> Optional[str]:
        """
        使用LLM生成预测
        
        Args:
            prompt: 输入提示词
            
        Returns:
            预测结果
        """
        try:
            # 生成参数
            generation_config = self.config['generation']
            
            # 生成文本
            outputs = self.pipeline(
                prompt,
                max_new_tokens=generation_config['max_new_tokens'],
                temperature=generation_config['temperature'],
                top_p=generation_config['top_p'],
                do_sample=generation_config['do_sample'],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # 提取生成的文本
            generated_text = outputs[0]['generated_text'].strip()
            
            # 清理输出，只保留预测值
            prediction = self._clean_prediction(generated_text)
            
            return prediction
            
        except Exception as e:
            print(f"生成预测时出错: {e}")
            return None
    
    def _clean_prediction(self, generated_text: str) -> str:
        """
        清理生成的预测文本，提取\boxed{}中的值
        
        Args:
            generated_text: 生成的文本
            
        Returns:
            清理后的预测值
        """
        import re
        
        # 移除多余的空格和换行
        cleaned = generated_text.strip()
        
        # 首先尝试提取\boxed{}中的内容
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_match = re.search(boxed_pattern, cleaned)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # 如果没有\boxed{}，尝试其他常见格式
        # 移除"预测结果："等前缀
        if "预测结果：" in cleaned:
            cleaned = cleaned.split("预测结果：")[-1].strip()
        
        # 移除引号
        cleaned = cleaned.strip('"').strip("'")
        
        # 移除可能的换行符和多余空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 如果结果为空，返回原始文本的第一行
        if not cleaned:
            cleaned = generated_text.split('\n')[0].strip()
        
        return cleaned
    
    def predict_single(self, features: Dict[str, Any], target_feature: str) -> Optional[str]:
        """
        对单个样本进行预测
        
        Args:
            features: 特征字典
            target_feature: 目标特征名称
            
        Returns:
            预测结果
        """
        if self.rag_retriever is None or self.pipeline is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # RAG检索相似样本
        similar_samples = self.rag_retriever.retrieve(pd.Series(features))
        
        # 构建提示词
        prompt = self.prompt_template.format_prompt(
            query_features=features,
            examples=similar_samples,
            target_feature=target_feature
        )
        
        # LLM生成预测
        prediction = self._generate_prediction(prompt)
        
        return prediction
    
    def show_prompt_template(self):
        """显示当前使用的提示词模板"""
        print("当前使用的提示词模板:")
        print("=" * 50)
        print(self.prompt_template.get_template())
        print("=" * 50)
