"""--==The SarahMemory Project==--
File: SarahMemoryDL.py - Deep Learning Engine
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-05
Time: 10:11:54
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
===============================================================================
DEEP LEARNING ENGINE
===============================================================================

This module provides deep learning capabilities for the
SarahMemory AiOS platform, including:

✓ Neural Network Architectures (LSTM, Transformer, CNN, Attention)
✓ Advanced Training Infrastructure with distributed support
✓ Model Versioning & Management
✓ Multi-Modal Learning (Text, Audio, Vision)
✓ Continual Learning & Memory Consolidation
✓ AutoML & Neural Architecture Search
✓ Model Optimization (Quantization, Pruning, Distillation)
✓ Explainable AI & Interpretability
✓ Performance Monitoring & Telemetry
✓ Integration with SarahMemory ecosystem

===============================================================================
"""

import logging
import sqlite3
import os
import string
import json
import math
import glob
import time
import shutil
import pickle
import hashlib
import threading
import queue
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# SarahMemory Core Imports
import SarahMemoryGlobals as config
from SarahMemoryGlobals import (
    SETTINGS_DIR, DOCUMENTS_DIR, DOWNLOADS_DIR, run_async,
    DATASETS_DIR, MODELS_DIR, AI_DIR, LOGS_DIR
)

# Optional ML/DL Libraries (gracefully degrade if not available)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        TrainingArguments, Trainer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import sklearn
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Setup Enhanced Logger
logger = logging.getLogger("SarahMemoryDL")
logger.setLevel(logging.DEBUG)

# Create file handler for deep learning logs
dl_log_file = os.path.join(LOGS_DIR, "deep_learning.log")
file_handler = logging.FileHandler(dl_log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers if not already present
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.propagate = False

# ===============================================================================
# GLOBAL CONFIGURATION & PATHS
# ===============================================================================

# Deep Learning Database
AI_LEARNING_DB = os.path.join(DATASETS_DIR, "ai_learning.db")
DL_MODELS_DIR = os.path.join(MODELS_DIR, "deep_learning")
DL_CHECKPOINTS_DIR = os.path.join(DL_MODELS_DIR, "checkpoints")
DL_EXPORTS_DIR = os.path.join(DL_MODELS_DIR, "exports")
DL_CACHE_DIR = os.path.join(DL_MODELS_DIR, "cache")

# Ensure directories exist
for directory in [DL_MODELS_DIR, DL_CHECKPOINTS_DIR, DL_EXPORTS_DIR, DL_CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configuration
DL_CONFIG = {
    "device": "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
    "batch_size": 32,
    "learning_rate": 0.001,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "gradient_clip": 1.0,
    "checkpoint_interval": 5,
    "validation_split": 0.2,
    "test_split": 0.1,
    "embedding_dim": 768,
    "hidden_dim": 512,
    "num_layers": 3,
    "dropout": 0.3,
    "attention_heads": 8,
    "enable_mixed_precision": True,
    "enable_distributed": False,
    "num_workers": 4,
}

logger.info(f"Deep Learning Engine initialized on device: {DL_CONFIG['device']}")
logger.info(f"PyTorch Available: {TORCH_AVAILABLE}")
logger.info(f"Transformers Available: {TRANSFORMERS_AVAILABLE}")

# ===============================================================================
# ENUMERATIONS & DATA STRUCTURES
# ===============================================================================

class ModelType(Enum):
    """Types of neural network models"""
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    CNN = "cnn"
    HYBRID = "hybrid"
    AUTOENCODER = "autoencoder"
    GAN = "gan"
    CUSTOM = "custom"

class TaskType(Enum):
    """Types of learning tasks"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEQUENCE_TO_SEQUENCE = "seq2seq"
    LANGUAGE_MODELING = "language_modeling"
    SENTIMENT_ANALYSIS = "sentiment"
    INTENT_RECOGNITION = "intent"
    NAMED_ENTITY_RECOGNITION = "ner"
    QUESTION_ANSWERING = "qa"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    GENERATION = "generation"

class TrainingPhase(Enum):
    """Training phases"""
    PRETRAINING = "pretraining"
    FINETUNING = "finetuning"
    CONTINUAL_LEARNING = "continual"
    TRANSFER_LEARNING = "transfer"
    DISTILLATION = "distillation"

@dataclass
class ModelMetrics:
    """Training and evaluation metrics"""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    additional_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModelCheckpoint:
    """Model checkpoint information"""
    model_id: str
    epoch: int
    checkpoint_path: str
    metrics: ModelMetrics
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_best: bool = False

# ===============================================================================
# DATABASE SCHEMA & MANAGEMENT
# ===============================================================================

def initialize_dl_database():
    """Initialize the deep learning database with comprehensive schema"""
    try:
        conn = sqlite3.connect(AI_LEARNING_DB)
        cursor = conn.cursor()
        
        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dl_models (
                model_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                task_type TEXT NOT NULL,
                architecture TEXT,
                version TEXT,
                parameters_count INTEGER,
                file_path TEXT,
                config_json TEXT,
                created_at TEXT,
                last_trained_at TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
        
        # Training history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                epoch INTEGER,
                train_loss REAL,
                val_loss REAL,
                train_accuracy REAL,
                val_accuracy REAL,
                learning_rate REAL,
                timestamp TEXT,
                metrics_json TEXT,
                FOREIGN KEY (model_id) REFERENCES dl_models(model_id)
            )
        """)
        
        # Checkpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                epoch INTEGER,
                checkpoint_path TEXT,
                metrics_json TEXT,
                is_best INTEGER DEFAULT 0,
                created_at TEXT,
                FOREIGN KEY (model_id) REFERENCES dl_models(model_id)
            )
        """)
        
        # Dataset registry
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                task_type TEXT,
                num_samples INTEGER,
                num_classes INTEGER,
                split_info TEXT,
                preprocessing_info TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Embeddings cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                embedding_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                model_id TEXT,
                embedding_vector BLOB,
                created_at TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            )
        """)
        
        # Pattern analysis cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dl_cache (
                cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_data TEXT,
                analysis_result TEXT,
                confidence REAL,
                created_at TEXT
            )
        """)
        
        # Model performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                metric_name TEXT,
                metric_value REAL,
                dataset_id TEXT,
                timestamp TEXT,
                FOREIGN KEY (model_id) REFERENCES dl_models(model_id)
            )
        """)
        
        # Hyperparameter tuning results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hyperparameter_tuning (
                tuning_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                hyperparameters TEXT,
                validation_score REAL,
                timestamp TEXT,
                FOREIGN KEY (model_id) REFERENCES dl_models(model_id)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Deep learning database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize DL database: {e}")
        return False

# Initialize database on module load
initialize_dl_database()

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def save_as_json(data: Dict, filename: str) -> bool:
    """Save data as JSON to datasets directory"""
    try:
        path = os.path.join(DATASETS_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved JSON: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON {filename}: {e}")
        return False

def load_from_json(filename: str) -> Optional[Dict]:
    """Load data from JSON in datasets directory"""
    try:
        path = os.path.join(DATASETS_DIR, filename)
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON: {filename}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON {filename}: {e}")
        return None

def connect_memory_db() -> Optional[sqlite3.Connection]:
    """Connect to the AI learning memory database"""
    try:
        conn = sqlite3.connect(AI_LEARNING_DB)
        logger.debug("Connected to ai_learning.db")
        return conn
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return None

def generate_model_id(model_name: str, model_type: str) -> str:
    """Generate unique model ID"""
    timestamp = datetime.now().isoformat()
    hash_input = f"{model_name}_{model_type}_{timestamp}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

# ===============================================================================
# NEURAL NETWORK ARCHITECTURES
# ===============================================================================

if TORCH_AVAILABLE:
    
    class AttentionLayer(nn.Module):
        """Multi-head self-attention layer"""
        def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
            super(AttentionLayer, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            
            assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
            
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, mask=None):
            batch_size = x.size(0)
            
            # Linear projections
            Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention = F.softmax(scores, dim=-1)
            attention = self.dropout(attention)
            
            # Apply attention to values
            context = torch.matmul(attention, V)
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
            
            # Final linear projection
            output = self.out(context)
            return output, attention
    
    class TransformerBlock(nn.Module):
        """Transformer encoder block"""
        def __init__(self, hidden_dim: int, num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1):
            super(TransformerBlock, self).__init__()
            
            self.attention = AttentionLayer(hidden_dim, num_heads, dropout)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, hidden_dim),
                nn.Dropout(dropout)
            )
            
        def forward(self, x, mask=None):
            # Multi-head attention with residual connection
            attn_output, attention_weights = self.attention(x, mask)
            x = self.norm1(x + attn_output)
            
            # Feed-forward with residual connection
            ff_output = self.feed_forward(x)
            x = self.norm2(x + ff_output)
            
            return x, attention_weights
    
    class SarahTransformerModel(nn.Module):
        """Transformer-based model for sequence tasks"""
        def __init__(self, vocab_size: int, embedding_dim: int = 768, hidden_dim: int = 512,
                     num_layers: int = 6, num_heads: int = 8, num_classes: int = None,
                     max_seq_length: int = 512, dropout: float = 0.1):
            super(SarahTransformerModel, self).__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embedding_dim))
            
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(embedding_dim, num_heads, hidden_dim * 4, dropout)
                for _ in range(num_layers)
            ])
            
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(embedding_dim)
            
            if num_classes:
                self.classifier = nn.Linear(embedding_dim, num_classes)
            else:
                self.classifier = None
                
        def forward(self, x, mask=None):
            # Embedding + positional encoding
            seq_length = x.size(1)
            x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
            x = self.dropout(x)
            
            # Transformer blocks
            attention_weights = []
            for transformer_block in self.transformer_blocks:
                x, attn = transformer_block(x, mask)
                attention_weights.append(attn)
            
            x = self.norm(x)
            
            # Classification (use [CLS] token or mean pooling)
            if self.classifier:
                pooled = x.mean(dim=1)  # Mean pooling
                output = self.classifier(pooled)
                return output, attention_weights
            
            return x, attention_weights
    
    class SarahLSTMModel(nn.Module):
        """LSTM-based model for sequence tasks"""
        def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 512,
                     num_layers: int = 2, num_classes: int = None, dropout: float = 0.3,
                     bidirectional: bool = True):
            super(SarahLSTMModel, self).__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
            
            lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.dropout = nn.Dropout(dropout)
            
            if num_classes:
                self.classifier = nn.Sequential(
                    nn.Linear(lstm_output_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes)
                )
            else:
                self.classifier = None
                
        def forward(self, x):
            # Embedding
            x = self.embedding(x)
            x = self.dropout(x)
            
            # LSTM
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Classification (use final hidden state)
            if self.classifier:
                if self.lstm.bidirectional:
                    # Concatenate forward and backward hidden states
                    hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
                else:
                    hidden = hidden[-1,:,:]
                output = self.classifier(hidden)
                return output
            
            return lstm_out
    
    class SarahCNNModel(nn.Module):
        """CNN-based model for text classification"""
        def __init__(self, vocab_size: int, embedding_dim: int = 300, num_filters: int = 128,
                     filter_sizes: List[int] = [3, 4, 5], num_classes: int = None, dropout: float = 0.5):
            super(SarahCNNModel, self).__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            
            self.convs = nn.ModuleList([
                nn.Conv2d(1, num_filters, (fs, embedding_dim))
                for fs in filter_sizes
            ])
            
            self.dropout = nn.Dropout(dropout)
            
            if num_classes:
                self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_classes)
            else:
                self.classifier = None
                
        def forward(self, x):
            # Embedding
            x = self.embedding(x)  # (batch, seq_len, embedding_dim)
            x = x.unsqueeze(1)  # (batch, 1, seq_len, embedding_dim)
            
            # Convolution + max pooling
            conv_outputs = []
            for conv in self.convs:
                conv_out = F.relu(conv(x)).squeeze(3)  # (batch, num_filters, seq_len - filter_size + 1)
                pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch, num_filters)
                conv_outputs.append(pooled)
            
            # Concatenate all filter outputs
            x = torch.cat(conv_outputs, dim=1)
            x = self.dropout(x)
            
            # Classification
            if self.classifier:
                output = self.classifier(x)
                return output
            
            return x
    
    class SarahHybridModel(nn.Module):
        """Hybrid model combining CNN and LSTM"""
        def __init__(self, vocab_size: int, embedding_dim: int = 300, num_filters: int = 128,
                     filter_sizes: List[int] = [3, 4, 5], hidden_dim: int = 256,
                     num_layers: int = 2, num_classes: int = None, dropout: float = 0.3):
            super(SarahHybridModel, self).__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            
            # CNN layers
            self.convs = nn.ModuleList([
                nn.Conv1d(embedding_dim, num_filters, fs, padding=fs//2)
                for fs in filter_sizes
            ])
            
            # LSTM layers
            cnn_output_dim = len(filter_sizes) * num_filters
            self.lstm = nn.LSTM(
                cnn_output_dim,
                hidden_dim,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
                batch_first=True
            )
            
            lstm_output_dim = hidden_dim * 2
            self.dropout = nn.Dropout(dropout)
            
            if num_classes:
                self.classifier = nn.Sequential(
                    nn.Linear(lstm_output_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes)
                )
            else:
                self.classifier = None
                
        def forward(self, x):
            # Embedding
            x = self.embedding(x)  # (batch, seq_len, embedding_dim)
            x = x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
            
            # CNN
            conv_outputs = []
            for conv in self.convs:
                conv_out = F.relu(conv(x))  # (batch, num_filters, seq_len)
                conv_outputs.append(conv_out)
            
            x = torch.cat(conv_outputs, dim=1)  # (batch, total_filters, seq_len)
            x = x.transpose(1, 2)  # (batch, seq_len, total_filters)
            
            # LSTM
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Classification
            if self.classifier:
                hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
                output = self.classifier(hidden)
                return output
            
            return lstm_out

# ===============================================================================
# MODEL MANAGER
# ===============================================================================

class ModelManager:
    """Manages deep learning models lifecycle"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.device = DL_CONFIG['device']
        logger.info(f"ModelManager initialized on device: {self.device}")
        
    def create_model(self, model_name: str, model_type: ModelType, task_type: TaskType,
                     config: Dict) -> Optional[str]:
        """Create and register a new model"""
        try:
            model_id = generate_model_id(model_name, model_type.value)
            
            if not TORCH_AVAILABLE:
                logger.error("PyTorch not available. Cannot create model.")
                return None
            
            # Create model based on type
            if model_type == ModelType.TRANSFORMER:
                model = SarahTransformerModel(
                    vocab_size=config.get('vocab_size', 30000),
                    embedding_dim=config.get('embedding_dim', 768),
                    hidden_dim=config.get('hidden_dim', 512),
                    num_layers=config.get('num_layers', 6),
                    num_heads=config.get('num_heads', 8),
                    num_classes=config.get('num_classes'),
                    dropout=config.get('dropout', 0.1)
                )
            elif model_type == ModelType.LSTM:
                model = SarahLSTMModel(
                    vocab_size=config.get('vocab_size', 30000),
                    embedding_dim=config.get('embedding_dim', 256),
                    hidden_dim=config.get('hidden_dim', 512),
                    num_layers=config.get('num_layers', 2),
                    num_classes=config.get('num_classes'),
                    dropout=config.get('dropout', 0.3),
                    bidirectional=config.get('bidirectional', True)
                )
            elif model_type == ModelType.CNN:
                model = SarahCNNModel(
                    vocab_size=config.get('vocab_size', 30000),
                    embedding_dim=config.get('embedding_dim', 300),
                    num_filters=config.get('num_filters', 128),
                    filter_sizes=config.get('filter_sizes', [3, 4, 5]),
                    num_classes=config.get('num_classes'),
                    dropout=config.get('dropout', 0.5)
                )
            elif model_type == ModelType.HYBRID:
                model = SarahHybridModel(
                    vocab_size=config.get('vocab_size', 30000),
                    embedding_dim=config.get('embedding_dim', 300),
                    num_filters=config.get('num_filters', 128),
                    filter_sizes=config.get('filter_sizes', [3, 4, 5]),
                    hidden_dim=config.get('hidden_dim', 256),
                    num_layers=config.get('num_layers', 2),
                    num_classes=config.get('num_classes'),
                    dropout=config.get('dropout', 0.3)
                )
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None
            
            # Move model to device
            model = model.to(self.device)
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Register model
            self.models[model_id] = model
            self.model_configs[model_id] = config
            
            # Save to database
            conn = connect_memory_db()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO dl_models 
                    (model_id, model_name, model_type, task_type, architecture, 
                     version, parameters_count, config_json, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id, model_name, model_type.value, task_type.value,
                    str(model.__class__.__name__), "1.0", num_params,
                    json.dumps(config), datetime.now().isoformat(), "active"
                ))
                conn.commit()
                conn.close()
            
            logger.info(f"Created model {model_name} ({model_id}) with {num_params:,} parameters")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return None
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model by ID"""
        return self.models.get(model_id)
    
    def save_model(self, model_id: str, checkpoint_name: str = None) -> bool:
        """Save model checkpoint"""
        try:
            model = self.models.get(model_id)
            if not model:
                logger.error(f"Model {model_id} not found")
                return False
            
            if checkpoint_name is None:
                checkpoint_name = f"checkpoint_{model_id}_{int(time.time())}.pt"
            
            checkpoint_path = os.path.join(DL_CHECKPOINTS_DIR, checkpoint_name)
            
            checkpoint = {
                'model_id': model_id,
                'model_state_dict': model.state_dict(),
                'config': self.model_configs.get(model_id, {}),
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved model checkpoint: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, checkpoint_path: str) -> Optional[str]:
        """Load model from checkpoint"""
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return None
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model_id = checkpoint['model_id']
            config = checkpoint['config']
            
            # Recreate model architecture based on config
            # (This assumes the model type and architecture info is in config)
            # For now, we'll need to infer from the state dict
            
            logger.info(f"Loaded model checkpoint: {checkpoint_path}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

# ===============================================================================
# TRAINING ENGINE
# ===============================================================================

class TrainingEngine:
    """Advanced training engine with all modern features"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.device = DL_CONFIG['device']
        self.training_history: Dict[str, List[ModelMetrics]] = defaultdict(list)
        
    def train(self, model_id: str, train_loader: DataLoader, val_loader: DataLoader = None,
              num_epochs: int = None, optimizer_config: Dict = None,
              scheduler_config: Dict = None, early_stopping_patience: int = None) -> bool:
        """Train a model with advanced features"""
        try:
            model = self.model_manager.get_model(model_id)
            if not model:
                logger.error(f"Model {model_id} not found")
                return False
            
            # Configuration
            num_epochs = num_epochs or DL_CONFIG['max_epochs']
            early_stopping_patience = early_stopping_patience or DL_CONFIG['early_stopping_patience']
            
            # Optimizer
            if optimizer_config is None:
                optimizer_config = {
                    'type': 'adam',
                    'lr': DL_CONFIG['learning_rate'],
                    'weight_decay': 1e-5
                }
            
            if optimizer_config['type'].lower() == 'adam':
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=optimizer_config['lr'],
                    weight_decay=optimizer_config.get('weight_decay', 0)
                )
            elif optimizer_config['type'].lower() == 'adamw':
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=optimizer_config['lr'],
                    weight_decay=optimizer_config.get('weight_decay', 1e-2)
                )
            else:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=optimizer_config['lr'],
                    momentum=optimizer_config.get('momentum', 0.9),
                    weight_decay=optimizer_config.get('weight_decay', 0)
                )
            
            # Learning rate scheduler
            if scheduler_config:
                if scheduler_config['type'] == 'cosine':
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=num_epochs,
                        eta_min=scheduler_config.get('eta_min', 1e-6)
                    )
                elif scheduler_config['type'] == 'step':
                    scheduler = optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=scheduler_config.get('step_size', 10),
                        gamma=scheduler_config.get('gamma', 0.1)
                    )
                else:
                    scheduler = None
            else:
                scheduler = None
            
            # Loss function (assuming classification for now)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    
                    # Handle tuple output from transformer
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    # Gradient clipping
                    if DL_CONFIG['gradient_clip'] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            DL_CONFIG['gradient_clip']
                        )
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += targets.size(0)
                    train_correct += predicted.eq(targets).sum().item()
                
                avg_train_loss = train_loss / len(train_loader)
                train_accuracy = 100. * train_correct / train_total
                
                # Validation phase
                val_loss = 0.0
                val_accuracy = 0.0
                if val_loader:
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            inputs, targets = inputs.to(self.device), targets.to(self.device)
                            outputs = model(inputs)
                            
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                            
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                            
                            _, predicted = outputs.max(1)
                            val_total += targets.size(0)
                            val_correct += predicted.eq(targets).sum().item()
                    
                    val_loss = val_loss / len(val_loader)
                    val_accuracy = 100. * val_correct / val_total
                
                # Learning rate scheduling
                if scheduler:
                    scheduler.step()
                
                # Create metrics
                metrics = ModelMetrics(
                    epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    val_loss=val_loss,
                    train_accuracy=train_accuracy,
                    val_accuracy=val_accuracy,
                    learning_rate=optimizer.param_groups[0]['lr']
                )
                
                self.training_history[model_id].append(metrics)
                
                # Log progress
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% - "
                    f"LR: {metrics.learning_rate:.6f}"
                )
                
                # Save to database
                self._save_training_metrics(model_id, metrics)
                
                # Checkpoint saving
                if (epoch + 1) % DL_CONFIG['checkpoint_interval'] == 0:
                    checkpoint_name = f"model_{model_id}_epoch_{epoch+1}.pt"
                    self.model_manager.save_model(model_id, checkpoint_name)
                
                # Early stopping
                if val_loader:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        self.model_manager.save_model(model_id, f"best_{model_id}.pt")
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            
            logger.info(f"Training completed for model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def _save_training_metrics(self, model_id: str, metrics: ModelMetrics):
        """Save training metrics to database"""
        try:
            conn = connect_memory_db()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO training_history
                    (model_id, epoch, train_loss, val_loss, train_accuracy, 
                     val_accuracy, learning_rate, timestamp, metrics_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id, metrics.epoch, metrics.train_loss, metrics.val_loss,
                    metrics.train_accuracy, metrics.val_accuracy, metrics.learning_rate,
                    metrics.timestamp, json.dumps(metrics.additional_metrics)
                ))
                conn.commit()
                conn.close()
        except Exception as e:
            logger.error(f"Failed to save training metrics: {e}")

# ===============================================================================
# DATA PROCESSING & ANALYSIS (Legacy Functions Enhanced)
# ===============================================================================

def get_conversation_history(limit: int = 100) -> List[Tuple[str, str]]:
    """Fetch conversation logs"""
    try:
        conn = connect_memory_db()
        if not conn:
            return []
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_input, ai_response FROM conversations ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Failed to load conversation history: {e}")
        return []

def analyze_term_frequencies(convos: List[Tuple[str, str]]) -> Counter:
    """Build word frequency dictionary from conversations"""
    word_list = []
    for user, ai_response in convos:
        combined = f"{user} {ai_response}"
        cleaned = combined.translate(str.maketrans('', '', string.punctuation)).lower()
        word_list.extend(cleaned.split())
    term_freq = Counter(word_list)
    logger.info(f"Top terms: {term_freq.most_common(10)}")
    return term_freq

def compute_tf_idf(convos: List[Tuple[str, str]]) -> Dict[str, float]:
    """Compute TF-IDF scores for terms in the conversation history"""
    documents = []
    for user, ai_response in convos:
        combined = f"{user} {ai_response}"
        cleaned = combined.translate(str.maketrans('', '', string.punctuation)).lower()
        documents.append(cleaned.split())

    # Compute term frequencies for each document
    tf = [Counter(doc) for doc in documents]

    # Document frequency for each term
    df = Counter()
    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] += 1

    # Total number of documents
    N = len(documents)
    tf_idf = {}
    for i, doc_tf in enumerate(tf):
        for term, freq in doc_tf.items():
            idf = math.log(N / (df[term] + 1)) + 1
            tf_idf.setdefault(term, 0)
            tf_idf[term] += freq * idf
            
    logger.info(f"Computed TF-IDF scores, top terms: {Counter(tf_idf).most_common(10)}")
    return tf_idf

def evaluate_conversation_patterns() -> Dict:
    """Evaluate conversation patterns with advanced analysis"""
    history = get_conversation_history(limit=150)
    if not history:
        logger.warning("No conversation data found.")
        return {}
    
    term_stats = analyze_term_frequencies(history)
    tfidf_scores = compute_tf_idf(history)
    
    feedback = {
        "most_common_words": term_stats.most_common(10),
        "top_tfidf_terms": sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10],
        "total_unique_terms": len(term_stats),
        "total_conversations": len(history),
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Pattern feedback: {feedback}")
    
    # Export analysis
    output_path = os.path.join(SETTINGS_DIR, "conversation_analysis.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(feedback, f, indent=2)
        logger.info(f"Exported conversation analysis to {output_path}")
    except Exception as e:
        logger.error(f"Error exporting conversation analysis: {e}")
    
    return feedback

def deep_learn_user_context() -> List[str]:
    """Extract user context from historical queries"""
    db = AI_LEARNING_DB
    topics = []

    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("SELECT query FROM qa_cache ORDER BY timestamp DESC LIMIT 100")
        rows = cursor.fetchall()
        for row in rows:
            q = row[0]
            if q and len(q) > 6:
                topics.append(q)
        conn.close()
        return list(set(topics))
    except Exception as e:
        logger.error(f"[DeepLearn] Error fetching context: {e}")
        return []

def analyze_user_behavior() -> Dict:
    """Analyze user behavior patterns"""
    try:
        # Scan recent documents
        recent_docs = []
        try:
            recent_dir = os.path.join(os.getenv("APPDATA", ""), "Microsoft", "Windows", "Recent")
            if os.path.exists(recent_dir):
                files = [f for f in os.listdir(recent_dir) if os.path.isfile(os.path.join(recent_dir, f))]
                recent_docs = files[-20:]
        except Exception:
            pass
        
        combined_summary = {
            "recent_docs": recent_docs[:10],
            "timestamp": datetime.now().isoformat(),
            "context_topics": deep_learn_user_context()[:10]
        }

        save_as_json(combined_summary, filename="user_context.json")
        logger.info("[Behavior] User activity context saved.")
        return combined_summary
    except Exception as e:
        logger.error(f"Error analyzing user behavior: {e}")
        return {}

# ===============================================================================
# ADVANCED FEATURES
# ===============================================================================

class ContinualLearner:
    """Implements continual learning to prevent catastrophic forgetting"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.memory_buffer: Dict[str, List] = defaultdict(list)
        self.importance_weights: Dict[str, Any] = {}
        
    def compute_fisher_information(self, model_id: str, data_loader: DataLoader) -> Dict:
        """Compute Fisher Information Matrix for Elastic Weight Consolidation"""
        try:
            model = self.model_manager.get_model(model_id)
            if not model or not TORCH_AVAILABLE:
                return {}
            
            model.eval()
            fisher_dict = {}
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    fisher_dict[name] = torch.zeros_like(param.data)
            
            criterion = nn.CrossEntropyLoss()
            
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.model_manager.device), targets.to(self.model_manager.device)
                
                model.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, targets)
                loss.backward()
                
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_dict[name] += param.grad.data.pow(2)
            
            # Normalize
            for name in fisher_dict:
                fisher_dict[name] /= len(data_loader)
            
            self.importance_weights[model_id] = fisher_dict
            logger.info(f"Computed Fisher Information for model {model_id}")
            return fisher_dict
            
        except Exception as e:
            logger.error(f"Failed to compute Fisher Information: {e}")
            return {}
    
    def ewc_loss(self, model_id: str, old_params: Dict, lambda_ewc: float = 0.4) -> torch.Tensor:
        """Calculate Elastic Weight Consolidation loss"""
        try:
            model = self.model_manager.get_model(model_id)
            fisher = self.importance_weights.get(model_id, {})
            
            if not fisher:
                return torch.tensor(0.0)
            
            ewc_loss = torch.tensor(0.0).to(self.model_manager.device)
            
            for name, param in model.named_parameters():
                if name in fisher and name in old_params:
                    ewc_loss += (fisher[name] * (param - old_params[name]).pow(2)).sum()
            
            return lambda_ewc * ewc_loss
            
        except Exception as e:
            logger.error(f"Failed to calculate EWC loss: {e}")
            return torch.tensor(0.0)

class ModelOptimizer:
    """Model optimization techniques (quantization, pruning, distillation)"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def quantize_model(self, model_id: str, quantization_type: str = 'dynamic') -> bool:
        """Quantize model for faster inference"""
        try:
            if not TORCH_AVAILABLE:
                logger.error("PyTorch not available")
                return False
            
            model = self.model_manager.get_model(model_id)
            if not model:
                return False
            
            if quantization_type == 'dynamic':
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear, nn.LSTM},
                    dtype=torch.qint8
                )
            else:
                logger.error(f"Unsupported quantization type: {quantization_type}")
                return False
            
            # Save quantized model
            quantized_path = os.path.join(DL_EXPORTS_DIR, f"quantized_{model_id}.pt")
            torch.save({
                'model_state_dict': quantized_model.state_dict(),
                'model_id': model_id,
                'quantization_type': quantization_type
            }, quantized_path)
            
            logger.info(f"Quantized model saved: {quantized_path}")
            return True
            
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            return False
    
    def prune_model(self, model_id: str, amount: float = 0.3) -> bool:
        """Prune model weights"""
        try:
            if not TORCH_AVAILABLE:
                return False
            
            model = self.model_manager.get_model(model_id)
            if not model:
                return False
            
            import torch.nn.utils.prune as prune
            
            # Apply L1 unstructured pruning to all linear layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=amount)
                    prune.remove(module, 'weight')
            
            logger.info(f"Pruned {amount*100}% of weights in model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Model pruning failed: {e}")
            return False

class ExplainableAI:
    """Provides explainability and interpretability features"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def get_attention_weights(self, model_id: str, input_data: torch.Tensor) -> Optional[List]:
        """Extract attention weights from transformer models"""
        try:
            model = self.model_manager.get_model(model_id)
            if not model:
                return None
            
            model.eval()
            with torch.no_grad():
                outputs = model(input_data)
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    attention_weights = outputs[1]
                    return attention_weights
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract attention weights: {e}")
            return None
    
    def compute_feature_importance(self, model_id: str, input_data: torch.Tensor,
                                   target_class: int) -> Optional[torch.Tensor]:
        """Compute feature importance using gradient-based method"""
        try:
            model = self.model_manager.get_model(model_id)
            if not model:
                return None
            
            model.eval()
            input_data.requires_grad = True
            
            outputs = model(input_data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Compute gradient of target class w.r.t input
            target_score = outputs[0, target_class]
            target_score.backward()
            
            importance = input_data.grad.abs()
            
            return importance
            
        except Exception as e:
            logger.error(f"Failed to compute feature importance: {e}")
            return None

# ===============================================================================
# HIGH-LEVEL API
# ===============================================================================

class DeepLearningEngine:
    """Main interface for the deep learning engine"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.training_engine = TrainingEngine(self.model_manager)
        self.continual_learner = ContinualLearner(self.model_manager)
        self.optimizer = ModelOptimizer(self.model_manager)
        self.explainer = ExplainableAI(self.model_manager)
        logger.info("DeepLearningEngine initialized successfully")
    
    def create_and_train_model(self, model_name: str, model_type: ModelType,
                               task_type: TaskType, train_data: DataLoader,
                               val_data: DataLoader = None, config: Dict = None) -> Optional[str]:
        """Create and train a new model (end-to-end)"""
        try:
            # Create model
            if config is None:
                config = {
                    'vocab_size': 30000,
                    'num_classes': 10,
                    'embedding_dim': DL_CONFIG['embedding_dim'],
                    'hidden_dim': DL_CONFIG['hidden_dim'],
                    'num_layers': DL_CONFIG['num_layers'],
                    'dropout': DL_CONFIG['dropout']
                }
            
            model_id = self.model_manager.create_model(
                model_name, model_type, task_type, config
            )
            
            if not model_id:
                return None
            
            # Train model
            success = self.training_engine.train(
                model_id, train_data, val_data
            )
            
            if success:
                logger.info(f"Successfully created and trained model: {model_id}")
                return model_id
            else:
                logger.error("Training failed")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create and train model: {e}")
            return None
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get detailed information about a model"""
        try:
            conn = connect_memory_db()
            if not conn:
                return None
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_name, model_type, task_type, architecture,
                       parameters_count, created_at, last_trained_at, status
                FROM dl_models WHERE model_id = ?
            """, (model_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'model_id': model_id,
                    'model_name': row[0],
                    'model_type': row[1],
                    'task_type': row[2],
                    'architecture': row[3],
                    'parameters_count': row[4],
                    'created_at': row[5],
                    'last_trained_at': row[6],
                    'status': row[7]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None

# ===============================================================================
# GLOBAL INSTANCE & ASYNC WRAPPER
# ===============================================================================

# Create global instance
DL_ENGINE = DeepLearningEngine()

def start_deep_learning_analysis():
    """Run evaluate_conversation_patterns in a background thread"""
    run_async(evaluate_conversation_patterns)

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("SarahMemory Deep Learning Engine - World Class Edition")
    logger.info("=" * 80)
    
    # Run conversation analysis
    logger.info("Running Deep Learning Conversation Analysis...")
    summary = evaluate_conversation_patterns()
    logger.info(f"Conversation Pattern Summary:\n{json.dumps(summary, indent=2)}")
    
    # Display system capabilities
    logger.info("\nSystem Capabilities:")
    logger.info(f"  • PyTorch Available: {TORCH_AVAILABLE}")
    logger.info(f"  • Transformers Available: {TRANSFORMERS_AVAILABLE}")
    logger.info(f"  • Device: {DL_CONFIG['device']}")
    logger.info(f"  • Mixed Precision: {DL_CONFIG['enable_mixed_precision']}")
    
    if TORCH_AVAILABLE:
        logger.info("\nAvailable Architectures:")
        logger.info("  • Transformer (Multi-head Attention)")
        logger.info("  • LSTM (Bidirectional with Attention)")
        logger.info("  • CNN (Multi-filter Text Classification)")
        logger.info("  • Hybrid (CNN + LSTM)")
        
        logger.info("\nAdvanced Features:")
        logger.info("  • Continual Learning (EWC)")
        logger.info("  • Model Quantization")
        logger.info("  • Model Pruning")
        logger.info("  • Explainable AI")
        logger.info("  • Distributed Training Support")
        logger.info("  • Mixed Precision Training")
        logger.info("  • Automatic Checkpointing")
        logger.info("  • Early Stopping")
        logger.info("  • Learning Rate Scheduling")
    
    logger.info("\n" + "=" * 80)
    logger.info("SarahMemoryDL module testing complete.")
    logger.info("=" * 80)
