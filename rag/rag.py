"""
RAG (Retrieval-Augmented Generation) 检索模块
用于从训练集中检索最相似的k个样本
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class RAGRetriever:
    """RAG检索器，用于从训练集中检索最相似的样本"""
    
    def __init__(self, k: int = 5, similarity_method: str = "cosine"):
        """
        初始化RAG检索器
        
        Args:
            k: 检索的样本数量
            similarity_method: 相似度计算方法 ("cosine", "euclidean", "manhattan")
        """
        self.k = k
        self.similarity_method = similarity_method
        self.scaler = StandardScaler()
        self.train_data = None
        self.train_features = None
        self.train_targets = None
        self.feature_columns = None
        
    def fit(self, train_df: pd.DataFrame, target_feature: str):
        """
        训练RAG检索器
        
        Args:
            train_df: 训练数据DataFrame
            target_feature: 目标特征列名
        """
        # 分离特征和目标
        self.feature_columns = [col for col in train_df.columns if col != target_feature]
        self.train_features = train_df[self.feature_columns].copy()
        self.train_targets = train_df[target_feature].copy()
        
        # 处理缺失值：用均值填充数值型，用众数填充分类型
        for col in self.feature_columns:
            if self.train_features[col].dtype in ['int64', 'float64']:
                # 数值型特征用均值填充
                self.train_features[col] = self.train_features[col].fillna(
                    self.train_features[col].mean()
                )
            else:
                # 分类型特征用众数填充
                mode_value = self.train_features[col].mode()
                if len(mode_value) > 0:
                    self.train_features[col] = self.train_features[col].fillna(mode_value[0])
                else:
                    # 如果没有众数，用第一个非空值填充
                    first_valid = self.train_features[col].dropna().iloc[0] if not self.train_features[col].dropna().empty else "Unknown"
                    self.train_features[col] = self.train_features[col].fillna(first_valid)
        
        # 对分类特征进行编码
        self._encode_categorical_features()
        
        # 标准化特征
        self.train_features_scaled = self.scaler.fit_transform(self.train_features)
        
        # 保存原始训练数据用于展示
        self.train_data = train_df.copy()
        
    def _encode_categorical_features(self):
        """对分类特征进行编码"""
        for col in self.feature_columns:
            if self.train_features[col].dtype == 'object':
                # 使用标签编码
                unique_values = self.train_features[col].unique()
                label_map = {val: idx for idx, val in enumerate(unique_values)}
                self.train_features[col] = self.train_features[col].map(label_map)
                # 保存编码映射
                if not hasattr(self, 'label_maps'):
                    self.label_maps = {}
                self.label_maps[col] = label_map
    
    def _preprocess_query(self, query_row: pd.Series) -> np.ndarray:
        """
        预处理查询样本
        
        Args:
            query_row: 查询样本的特征
            
        Returns:
            预处理后的特征向量
        """
        query_features = query_row[self.feature_columns].copy()
        
        # 处理缺失值
        for col in self.feature_columns:
            if query_features[col].isna():
                if col in self.train_features.columns:
                    if self.train_features[col].dtype in ['int64', 'float64']:
                        # 数值型特征用训练集均值填充
                        query_features[col] = self.train_features[col].mean()
                    else:
                        # 分类型特征用训练集众数填充
                        mode_value = self.train_features[col].mode()
                        if len(mode_value) > 0:
                            query_features[col] = mode_value[0]
                        else:
                            query_features[col] = self.train_features[col].iloc[0]
                else:
                    query_features[col] = 0
        
        # 对分类特征进行编码
        for col in self.feature_columns:
            if col in self.label_maps:
                if query_features[col] not in self.label_maps[col]:
                    # 如果查询样本的值不在训练集中，使用最常见的编码
                    query_features[col] = 0
                else:
                    query_features[col] = self.label_maps[col][query_features[col]]
        
        # 标准化
        query_scaled = self.scaler.transform([query_features])
        return query_scaled[0]
    
    def _calculate_similarity(self, query_vector: np.ndarray, train_vectors: np.ndarray) -> np.ndarray:
        """
        计算相似度
        
        Args:
            query_vector: 查询向量
            train_vectors: 训练向量矩阵
            
        Returns:
            相似度分数数组
        """
        if self.similarity_method == "cosine":
            similarities = cosine_similarity([query_vector], train_vectors)[0]
        elif self.similarity_method == "euclidean":
            distances = np.linalg.norm(train_vectors - query_vector, axis=1)
            similarities = 1 / (1 + distances)  # 转换为相似度
        elif self.similarity_method == "manhattan":
            distances = np.sum(np.abs(train_vectors - query_vector), axis=1)
            similarities = 1 / (1 + distances)  # 转换为相似度
        else:
            raise ValueError(f"Unsupported similarity method: {self.similarity_method}")
        
        return similarities
    
    def retrieve(self, query_row: pd.Series) -> List[Dict[str, Any]]:
        """
        检索最相似的k个样本
        
        Args:
            query_row: 查询样本
            
        Returns:
            最相似的k个样本列表，每个样本包含特征和目标值
        """
        # 预处理查询样本
        query_vector = self._preprocess_query(query_row)
        
        # 计算相似度
        similarities = self._calculate_similarity(query_vector, self.train_features_scaled)
        
        # 获取最相似的k个样本的索引
        top_k_indices = np.argsort(similarities)[-self.k:][::-1]  # 降序排列
        
        # 构建结果
        retrieved_samples = []
        for idx in top_k_indices:
            sample = {
                'features': self.train_data.iloc[idx][self.feature_columns].to_dict(),
                'target': self.train_targets.iloc[idx],
                'similarity': similarities[idx]
            }
            retrieved_samples.append(sample)
        
        return retrieved_samples
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        获取特征信息
        
        Returns:
            特征信息字典
        """
        return {
            'feature_columns': self.feature_columns,
            'num_features': len(self.feature_columns),
            'num_samples': len(self.train_data) if self.train_data is not None else 0
        }
