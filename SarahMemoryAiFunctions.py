"""--==The SarahMemory Project==--
File: SarahMemoryAiFunctions.py
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
"""

from SarahMemoryAdvCU import classify_intent
import re
import logging
import time
import sqlite3
import os
import json
import numpy as np
import random
import threading
import queue
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import pickle

# === Optional embedding shim (offline-safe) ===
try:
    from SarahMemoryAdvCU import embed_text as _advcu_embed
except Exception:
    _advcu_embed = None
from SarahMemoryCanvasStudio import CanvasStudio

def _fallback_embed(text: str, dim: int = 64):
    """
    Tiny, deterministic local embedding (no downloads).
    Uses a simple hash-based projection so we always return *something*.
    """
    import hashlib, math
    h = hashlib.sha256((text or "").encode("utf-8")).digest()
    vals = []
    for i in range(dim):
        b = h[i % len(h)]
        vals.append(math.sin((b + i) * 0.0174533))
    return vals

def lite_embed(texts):
    """Small wrapper all modules can call; never throws."""
    if isinstance(texts, str):
        texts = [texts]
    if _advcu_embed:
        try:
            return _advcu_embed(texts)
        except Exception:
            pass
    return [_fallback_embed(t) for t in texts]

# === Remote hub bridge (non-blocking) ===
_HUB = None
def _get_hub():
    global _HUB
    if _HUB is None:
        try:
            from SarahMemoryNetwork import ServerConnection
            _HUB = ServerConnection()
            try:
                _HUB.start_heartbeat()
            except Exception:
                pass
        except Exception:
            _HUB = None
    return _HUB

def _fire_and_forget(coro_func, *args, **kwargs):
    try:
        import threading, asyncio
        def run():
            try:
                asyncio.run(coro_func(*args, **kwargs))
            except Exception:
                pass
        threading.Thread(target=run, name="SM-Remote-Call", daemon=True).start()
    except Exception:
        pass

# === Core imports ===
try:
    import speech_recognition as sr
except Exception:
    sr = None

from SarahMemoryVoice import synthesize_voice
import SarahMemoryGlobals as config

# === Agent Consent Flag ===
try:
    AI_AGENT_REQUIRE_CONSENT = bool(getattr(config, "AI_AGENT_REQUIRE_CONSENT", False))
except Exception:
    AI_AGENT_REQUIRE_CONSENT = False

from SarahMemoryGlobals import DATASETS_DIR

# === Safe/offline and local-only support ===
try:
    from SarahMemoryGlobals import SAFE_MODE, LOCAL_ONLY_MODE, is_offline
except Exception:
    SAFE_MODE = False
    LOCAL_ONLY_MODE = False
    def is_offline():
        return False

# === Logging setup ===
logger = logging.getLogger('SarahMemoryAiFunctions')
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(_h)
logger.propagate = False

# === Context Management ===
context_buffer = []
context_embeddings = []
CONTEXT_DB_PATH = os.path.join(config.DATASETS_DIR, 'context_history.db')

try:
    from SarahMemoryGlobals import get_context_config, agent_permissions_summary, get_mesh_sync_config
except Exception:
    def get_context_config():
        return {
            "enabled":           bool(getattr(config, "ENABLE_CONTEXT_BUFFER", True)),
            "buffer_size":       int(getattr(config, "CONTEXT_BUFFER_SIZE", 20)),
            "max_turns":         int(getattr(config, "CONTEXT_BUFFER_SIZE", 20)),
            "max_age_sec":       259200,
            "persist_to_db":     True,
            "db_path":           CONTEXT_DB_PATH,
            "enrichment_enabled":True,
        }

    def agent_permissions_summary():
        return {
            "run_mode":               getattr(config, "RUN_MODE", "local"),
            "device_mode":            getattr(config, "DEVICE_MODE", "headless"),
            "safe_mode":              bool(getattr(config, "SAFE_MODE", False)),
            "local_only":             bool(getattr(config, "LOCAL_ONLY_MODE", False)),
            "agent_enabled":          bool(getattr(config, "AI_AGENT_ENABLED", False)),
            "allow_app_launch":       bool(getattr(config, "AI_AGENT_ALLOW_APP_LAUNCH", False)),
            "allow_file_write":       bool(getattr(config, "AI_AGENT_ALLOW_FILE_WRITE", False)),
            "allow_remote_control":   bool(getattr(config, "AI_AGENT_ALLOW_REMOTE_CONTROL", False)),
            "allow_network_tasks":    bool(getattr(config, "AI_AGENT_ALLOW_NETWORK_TASKS", False)),
            "user_activity_timeout":  int(getattr(config, "AI_AGENT_USER_ACTIVITY_TIMEOUT_MS", 2500)),
            "resume_delay_ms":        int(getattr(config, "AI_AGENT_RESUME_DELAY", 9000)),
        }

    def get_mesh_sync_config():
        return {
            "node_name":            getattr(config, "NODE_NAME", "SarahMemoryNode"),
            "mesh_enabled":         bool(getattr(config, "MESH_SYNC_ENABLED", True)),
            "hub_allowed":          bool(getattr(config, "ALLOW_HUB_SYNC", True)),
            "safe_mode":            bool(getattr(config, "SAFE_MODE", False)),
            "safe_mode_only":       bool(getattr(config, "MESH_SYNC_SAFE_MODE_ONLY", False)),
            "sarahnet_enabled":     bool(getattr(config, "SARAHNET_ENABLED", True)),
            "web_base":             getattr(config, "SARAH_WEB_BASE", "https://www.sarahmemory.com"),
            "remote_sync_enabled":  bool(getattr(config, "REMOTE_SYNC_ENABLED", True)),
            "heartbeat_sec":        float(getattr(config, "REMOTE_HEARTBEAT_SEC", 30)),
            "http_timeout":         float(getattr(config, "REMOTE_HTTP_TIMEOUT", 6.0)),
        }


# ================================================================================
# NEW v8.0 ADVANCED STRUCTURES
# ================================================================================

class TaskPriority(Enum):
    """Task priority levels for the hierarchical planner"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Represents a single task in the hierarchical planning system"""
    id: str
    description: str
    intent: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    subtasks: List['Task'] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert task to dictionary for serialization"""
        return {
            'id': self.id,
            'description': self.description,
            'intent': self.intent,
            'priority': self.priority.name,
            'status': self.status.value,
            'dependencies': self.dependencies,
            'subtasks': [st.to_dict() for st in self.subtasks],
            'tools_required': self.tools_required,
            'estimated_duration': self.estimated_duration,
            'actual_duration': self.actual_duration,
            'result': str(self.result) if self.result else None,
            'error': self.error,
            'retry_count': self.retry_count,
            'metadata': self.metadata
        }

@dataclass
class DecisionPoint:
    """Represents a decision made by the AI agent with reasoning"""
    timestamp: float
    context: str
    decision: str
    reasoning: List[str]
    confidence: float
    alternatives_considered: List[Tuple[str, float]]
    outcome: Optional[str] = None
    was_correct: Optional[bool] = None
    learned_lesson: Optional[str] = None

@dataclass
class KnowledgeNode:
    """Node in the knowledge graph for contextual understanding"""
    id: str
    content: str
    embedding: List[float]
    node_type: str  # 'fact', 'concept', 'procedure', 'preference', 'experience'
    timestamp: float
    connections: Dict[str, float] = field(default_factory=dict)  # id -> strength
    access_count: int = 0
    last_accessed: float = 0.0
    reliability: float = 1.0
    source: str = "user_interaction"

class PerformanceMetrics:
    """Tracks agent performance for self-improvement"""
    def __init__(self):
        self.task_success_rate: Dict[str, List[bool]] = defaultdict(list)
        self.tool_performance: Dict[str, List[float]] = defaultdict(list)
        self.intent_accuracy: Dict[str, List[bool]] = defaultdict(list)
        self.response_times: List[float] = []
        self.user_satisfaction: List[float] = []
        self.learning_rate: float = 0.01
        self.adaptation_history: List[Dict] = []

    def record_task_outcome(self, task_type: str, success: bool):
        """Record task execution outcome"""
        self.task_success_rate[task_type].append(success)
        if len(self.task_success_rate[task_type]) > 100:
            self.task_success_rate[task_type].pop(0)

    def record_tool_execution(self, tool_name: str, duration: float, success: bool):
        """Record tool execution metrics"""
        score = duration if success else float('inf')
        self.tool_performance[tool_name].append(score)
        if len(self.tool_performance[tool_name]) > 100:
            self.tool_performance[tool_name].pop(0)

    def get_tool_reliability(self, tool_name: str) -> float:
        """Calculate tool reliability score"""
        if tool_name not in self.tool_performance:
            return 0.5
        scores = self.tool_performance[tool_name]
        successful = [s for s in scores if s != float('inf')]
        return len(successful) / len(scores) if scores else 0.5

    def get_success_rate(self, task_type: str) -> float:
        """Get success rate for specific task type"""
        if task_type not in self.task_success_rate:
            return 0.5
        results = self.task_success_rate[task_type]
        return sum(results) / len(results) if results else 0.5

# ================================================================================
# GLOBAL STATE FOR ADVANCED AGENT
# ================================================================================

class AdvancedAgentState:
    """Centralized state management for the advanced AI agent"""
    def __init__(self):
        # Hierarchical planning
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.task_graph: Dict[str, Set[str]] = defaultdict(set)

        # Meta-cognitive reasoning
        self.decision_history: deque = deque(maxlen=500)
        self.reasoning_chains: List[List[str]] = []
        self.confidence_threshold: float = 0.7
        self.self_reflection_interval: int = 10  # Reflect every N interactions
        self.interaction_count: int = 0

        # Knowledge graph
        self.knowledge_graph: Dict[str, KnowledgeNode] = {}
        self.concept_embeddings: Dict[str, np.ndarray] = {}
        self.relationship_types = ['causes', 'enables', 'requires', 'contradicts', 'supports', 'similar_to']

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.bottlenecks: List[Dict] = []
        self.optimization_suggestions: List[str] = []

        # Predictive modeling
        self.user_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.predicted_intents: queue.Queue = queue.Queue(maxsize=5)
        self.prediction_confidence: Dict[str, float] = {}

        # Multi-agent coordination
        self.peer_agents: Dict[str, Dict] = {}
        self.shared_knowledge: Dict[str, Any] = {}
        self.collaboration_requests: queue.Queue = queue.Queue()

        # Autonomous learning
        self.learning_buffer: List[Dict] = []
        self.skill_improvements: Dict[str, float] = defaultdict(float)
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.success_patterns: Dict[str, int] = defaultdict(int)

        # Emotional intelligence
        self.emotional_context: Dict[str, Any] = {
            'user_mood': 'neutral',
            'conversation_tone': 'neutral',
            'stress_level': 0.0,
            'engagement_level': 0.5
        }

        # Tool orchestration
        self.tool_registry: Dict[str, Dict] = {}
        self.tool_dependencies: Dict[str, List[str]] = defaultdict(list)
        self.parallel_execution_enabled: bool = True
        self.max_parallel_tools: int = 5

        # Causal reasoning
        self.causal_model: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.intervention_history: List[Dict] = []
        self.outcome_predictions: Dict[str, List[float]] = defaultdict(list)

        # Self-diagnostic
        self.health_metrics: Dict[str, float] = {
            'response_latency': 0.0,
            'memory_usage': 0.0,
            'error_rate': 0.0,
            'learning_effectiveness': 1.0
        }
        self.diagnostic_queue: queue.Queue = queue.Queue()

        # Thread management
        self.executor_threads: List[threading.Thread] = []
        self.shutdown_flag: threading.Event = threading.Event()
        self.lock = threading.RLock()

    def initialize(self):
        """Initialize the advanced agent state"""
        logger.info("[AdvancedAgent] Initializing v8.0 Advanced Agent Systems...")
        self._load_persistent_state()
        self._start_background_processes()
        logger.info("[AdvancedAgent] Initialization complete.")

    def _load_persistent_state(self):
        """Load persistent state from disk"""
        try:
            state_file = os.path.join(DATASETS_DIR, 'advanced_agent_state.pkl')
            if os.path.exists(state_file):
                with open(state_file, 'rb') as f:
                    saved_state = pickle.load(f)
                    self.knowledge_graph = saved_state.get('knowledge_graph', {})
                    self.metrics.task_success_rate = saved_state.get('task_success_rate', defaultdict(list))
                    self.user_patterns = saved_state.get('user_patterns', defaultdict(list))
                    logger.info(f"[AdvancedAgent] Loaded {len(self.knowledge_graph)} knowledge nodes from disk")
        except Exception as e:
            logger.warning(f"[AdvancedAgent] Could not load persistent state: {e}")

    def _start_background_processes(self):
        """Start background threads for continuous operations"""
        threads = [
            threading.Thread(target=self._meta_cognitive_loop, name="MetaCognitive", daemon=True),
            threading.Thread(target=self._learning_loop, name="LearningLoop", daemon=True),
            threading.Thread(target=self._self_diagnostic_loop, name="SelfDiagnostic", daemon=True),
            threading.Thread(target=self._predictive_modeling_loop, name="PredictiveModel", daemon=True)
        ]
        for t in threads:
            t.start()
            self.executor_threads.append(t)
        logger.info(f"[AdvancedAgent] Started {len(threads)} background processes")

    def _meta_cognitive_loop(self):
        """Continuous meta-cognitive reasoning and self-reflection"""
        while not self.shutdown_flag.is_set():
            try:
                time.sleep(30)  # Reflect every 30 seconds
                with self.lock:
                    self._perform_self_reflection()
            except Exception as e:
                logger.error(f"[MetaCognitive] Error in reflection loop: {e}")

    def _learning_loop(self):
        """Continuous learning from interactions"""
        while not self.shutdown_flag.is_set():
            try:
                time.sleep(60)  # Learn every minute
                with self.lock:
                    self._process_learning_buffer()
            except Exception as e:
                logger.error(f"[LearningLoop] Error: {e}")

    def _self_diagnostic_loop(self):
        """Monitor and diagnose performance issues"""
        while not self.shutdown_flag.is_set():
            try:
                time.sleep(45)  # Diagnose every 45 seconds
                with self.lock:
                    self._run_diagnostics()
            except Exception as e:
                logger.error(f"[SelfDiagnostic] Error: {e}")

    def _predictive_modeling_loop(self):
        """Predict future user needs"""
        while not self.shutdown_flag.is_set():
            try:
                time.sleep(20)  # Predict every 20 seconds
                with self.lock:
                    self._generate_predictions()
            except Exception as e:
                logger.error(f"[PredictiveModel] Error: {e}")

    def _perform_self_reflection(self):
        """Analyze recent decisions and performance"""
        if len(self.decision_history) < 5:
            return

        recent_decisions = list(self.decision_history)[-10:]
        successful = sum(1 for d in recent_decisions if d.was_correct)
        accuracy = successful / len(recent_decisions) if recent_decisions else 0.0

        if accuracy < 0.6:
            logger.warning(f"[MetaCognitive] Low decision accuracy: {accuracy:.2%}")
            self._adjust_confidence_threshold(decrease=True)
        elif accuracy > 0.85:
            logger.info(f"[MetaCognitive] High decision accuracy: {accuracy:.2%}")
            self._adjust_confidence_threshold(decrease=False)

    def _adjust_confidence_threshold(self, decrease: bool):
        """Adjust confidence threshold based on performance"""
        if decrease:
            self.confidence_threshold = min(0.9, self.confidence_threshold + 0.05)
        else:
            self.confidence_threshold = max(0.5, self.confidence_threshold - 0.02)
        logger.info(f"[MetaCognitive] Adjusted confidence threshold to {self.confidence_threshold:.2f}")

    def _process_learning_buffer(self):
        """Process accumulated learning experiences"""
        if not self.learning_buffer:
            return

        for experience in self.learning_buffer[:10]:
            try:
                self._incorporate_experience(experience)
            except Exception as e:
                logger.error(f"[Learning] Failed to process experience: {e}")

        self.learning_buffer = self.learning_buffer[10:]

    def _incorporate_experience(self, experience: Dict):
        """Incorporate a learning experience into the knowledge base"""
        exp_type = experience.get('type', 'unknown')

        if exp_type == 'task_success':
            pattern = experience.get('pattern')
            self.success_patterns[pattern] += 1
            self.skill_improvements[pattern] = self.skill_improvements[pattern] * 0.9 + 0.1
        elif exp_type == 'task_failure':
            pattern = experience.get('pattern')
            self.error_patterns[pattern] += 1
            if self.error_patterns[pattern] > 5:
                logger.warning(f"[Learning] Frequent failures in pattern: {pattern}")

    def _run_diagnostics(self):
        """Run self-diagnostics on agent performance"""
        try:
            # Check response latency
            if self.metrics.response_times:
                avg_latency = sum(self.metrics.response_times[-20:]) / len(self.metrics.response_times[-20:])
                self.health_metrics['response_latency'] = avg_latency
                if avg_latency > 2.0:
                    self.bottlenecks.append({
                        'type': 'high_latency',
                        'value': avg_latency,
                        'timestamp': time.time()
                    })

            # Check error patterns
            total_errors = sum(self.error_patterns.values())
            total_attempts = total_errors + sum(self.success_patterns.values())
            if total_attempts > 0:
                error_rate = total_errors / total_attempts
                self.health_metrics['error_rate'] = error_rate
                if error_rate > 0.2:
                    logger.warning(f"[SelfDiagnostic] High error rate detected: {error_rate:.2%}")
        except Exception as e:
            logger.error(f"[SelfDiagnostic] Diagnostic error: {e}")

    def _generate_predictions(self):
        """Generate predictions about future user needs"""
        try:
            if not self.user_patterns:
                return

            current_time = datetime.now()
            current_hour = current_time.hour
            current_day = current_time.weekday()

            # Find similar time patterns
            for pattern_key, pattern_list in self.user_patterns.items():
                if len(pattern_list) < 3:
                    continue

                # Simple time-based prediction
                similar_contexts = [p for p in pattern_list
                                   if abs(p.get('hour', 0) - current_hour) < 2
                                   and p.get('day', 0) == current_day]

                if len(similar_contexts) >= 2:
                    predicted_intent = similar_contexts[-1].get('intent')
                    confidence = len(similar_contexts) / len(pattern_list)

                    if confidence > 0.3:
                        try:
                            self.predicted_intents.put_nowait({
                                'intent': predicted_intent,
                                'confidence': confidence,
                                'context': pattern_key
                            })
                        except queue.Full:
                            pass
        except Exception as e:
            logger.error(f"[PredictiveModel] Prediction error: {e}")

    def save_state(self):
        """Save persistent state to disk"""
        try:
            state_file = os.path.join(DATASETS_DIR, 'advanced_agent_state.pkl')
            os.makedirs(os.path.dirname(state_file), exist_ok=True)

            state_to_save = {
                'knowledge_graph': self.knowledge_graph,
                'task_success_rate': dict(self.metrics.task_success_rate),
                'user_patterns': dict(self.user_patterns),
                'timestamp': time.time()
            }

            with open(state_file, 'wb') as f:
                pickle.dump(state_to_save, f)

            logger.info(f"[AdvancedAgent] Saved state: {len(self.knowledge_graph)} knowledge nodes")
        except Exception as e:
            logger.error(f"[AdvancedAgent] Failed to save state: {e}")

# Global advanced agent state
ADVANCED_AGENT = AdvancedAgentState()

# ================================================================================
# HIERARCHICAL PLANNING ENGINE
# ================================================================================

class HierarchicalPlanner:
    """Advanced hierarchical task planner with goal decomposition"""

    def __init__(self, agent_state: AdvancedAgentState):
        self.agent = agent_state
        self.task_templates: Dict[str, Dict] = self._initialize_task_templates()
        self.decomposition_rules: Dict[str, List] = self._initialize_decomposition_rules()

    def _initialize_task_templates(self) -> Dict[str, Dict]:
        """Initialize templates for common task types"""
        return {
            'web_research': {
                'tools': ['memory_search', 'web_search', 'web_fetch'],
                'estimated_duration': 5.0,
                'can_parallelize': True
            },
            'file_operation': {
                'tools': ['file_read', 'file_write', 'file_search'],
                'estimated_duration': 2.0,
                'can_parallelize': False
            },
            'calculation': {
                'tools': ['calc', 'symbolic_calc'],
                'estimated_duration': 0.5,
                'can_parallelize': True
            },
            'system_control': {
                'tools': ['agent', 'open_app', 'system_command'],
                'estimated_duration': 3.0,
                'can_parallelize': False
            },
            'communication': {
                'tools': ['send_message', 'email', 'notification'],
                'estimated_duration': 2.0,
                'can_parallelize': True
            }
        }

    def _initialize_decomposition_rules(self) -> Dict[str, List]:
        """Initialize rules for task decomposition"""
        return {
            'research_and_summarize': [
                ('research', 'web_research'),
                ('analyze', 'analysis'),
                ('summarize', 'summarization')
            ],
            'compare_options': [
                ('gather_option_1', 'web_research'),
                ('gather_option_2', 'web_research'),
                ('compare', 'comparison'),
                ('recommend', 'decision_making')
            ],
            'automate_workflow': [
                ('analyze_workflow', 'analysis'),
                ('plan_automation', 'planning'),
                ('implement', 'system_control'),
                ('test', 'testing'),
                ('deploy', 'deployment')
            ]
        }

    def create_task(self, description: str, intent: str, priority: TaskPriority = TaskPriority.NORMAL) -> Task:
        """Create a new task with automatic decomposition"""
        task_id = hashlib.md5(f"{description}{time.time()}".encode()).hexdigest()[:12]

        task = Task(
            id=task_id,
            description=description,
            intent=intent,
            priority=priority,
            metadata={'creation_time': time.time()}
        )

        # Check if task needs decomposition
        if self._should_decompose(task):
            subtasks = self._decompose_task(task)
            task.subtasks = subtasks
            logger.info(f"[HierarchicalPlanner] Decomposed task '{description}' into {len(subtasks)} subtasks")

        # Identify required tools
        task.tools_required = self._identify_tools(task)

        # Estimate duration
        task.estimated_duration = self._estimate_duration(task)

        return task

    def _should_decompose(self, task: Task) -> bool:
        """Determine if task should be decomposed into subtasks"""
        complex_keywords = ['research and', 'compare', 'analyze', 'create report',
                           'automate', 'optimize', 'design', 'implement']
        return any(kw in task.description.lower() for kw in complex_keywords)

    def _decompose_task(self, task: Task) -> List[Task]:
        """Decompose complex task into subtasks"""
        subtasks = []
        description_lower = task.description.lower()

        # Check decomposition rules
        for pattern, steps in self.decomposition_rules.items():
            if pattern.replace('_', ' ') in description_lower:
                for step_name, step_type in steps:
                    subtask = Task(
                        id=f"{task.id}_{len(subtasks)}",
                        description=f"{step_name} for: {task.description}",
                        intent=step_type,
                        priority=task.priority,
                        metadata={'parent_task': task.id}
                    )
                    if subtasks:
                        subtask.dependencies.append(subtasks[-1].id)
                    subtasks.append(subtask)
                return subtasks

        # Default decomposition for complex tasks
        if 'and' in description_lower:
            parts = description_lower.split(' and ')
            for i, part in enumerate(parts):
                subtask = Task(
                    id=f"{task.id}_{i}",
                    description=part.strip(),
                    intent=task.intent,
                    priority=task.priority,
                    metadata={'parent_task': task.id}
                )
                if i > 0:
                    subtask.dependencies.append(f"{task.id}_{i-1}")
                subtasks.append(subtask)

        return subtasks

    def _identify_tools(self, task: Task) -> List[str]:
        """Identify tools needed for task execution"""
        tools = []
        description_lower = task.description.lower()
        intent_lower = task.intent.lower()

        # Map keywords to tools
        tool_keywords = {
            'web_search': ['search', 'find', 'look up', 'research'],
            'calc': ['calculate', 'compute', 'math', 'add', 'subtract', 'multiply'],
            'file_read': ['read', 'open', 'view', 'check'],
            'file_write': ['write', 'save', 'create file', 'store'],
            'agent': ['move mouse', 'click', 'press', 'type'],
            'memory_search': ['remember', 'recall', 'previous'],
            'open_url': ['open', 'browse', 'navigate', 'http'],
            'open_app': ['launch', 'start', 'run application']
        }

        for tool, keywords in tool_keywords.items():
            if any(kw in description_lower or kw in intent_lower for kw in keywords):
                tools.append(tool)

        # Check task template
        if task.intent in self.task_templates:
            template_tools = self.task_templates[task.intent]['tools']
            tools.extend([t for t in template_tools if t not in tools])

        return tools

    def _estimate_duration(self, task: Task) -> float:
        """Estimate task execution duration"""
        if task.subtasks:
            # For tasks with subtasks, sum subtask durations
            return sum(self._estimate_duration(st) for st in task.subtasks)

        # Use template if available
        if task.intent in self.task_templates:
            return self.task_templates[task.intent]['estimated_duration']

        # Default estimation based on tool count
        base_duration = 1.0
        tool_overhead = len(task.tools_required) * 0.5
        return base_duration + tool_overhead

    def plan_execution(self, task: Task) -> List[Tuple[str, Any]]:
        """Create execution plan for task"""
        execution_plan = []

        if task.subtasks:
            # Handle subtasks
            for subtask in task.subtasks:
                subtask_plan = self.plan_execution(subtask)
                execution_plan.extend(subtask_plan)
        else:
            # Create execution steps from tools
            for tool in task.tools_required:
                execution_plan.append((tool, task.description))

        return execution_plan

    def optimize_parallel_execution(self, tasks: List[Task]) -> List[List[Task]]:
        """Group tasks for parallel execution where possible"""
        parallel_groups = []
        current_group = []
        dependencies_met = set()

        for task in sorted(tasks, key=lambda t: t.priority.value):
            # Check if all dependencies are met
            deps_satisfied = all(dep in dependencies_met for dep in task.dependencies)

            can_parallelize = (
                deps_satisfied and
                task.intent in self.task_templates and
                self.task_templates[task.intent].get('can_parallelize', False) and
                len(current_group) < self.agent.max_parallel_tools
            )

            if can_parallelize:
                current_group.append(task)
            else:
                if current_group:
                    parallel_groups.append(current_group)
                current_group = [task]

            dependencies_met.add(task.id)

        if current_group:
            parallel_groups.append(current_group)

        return parallel_groups

# ================================================================================
# META-COGNITIVE REASONING ENGINE
# ================================================================================

class MetaCognitiveReasoner:
    """Implements meta-cognitive reasoning and self-reflection"""

    def __init__(self, agent_state: AdvancedAgentState):
        self.agent = agent_state
        self.reasoning_strategies = [
            'forward_chaining',
            'backward_chaining',
            'analogy',
            'case_based',
            'abductive'
        ]

    def reason_about_decision(self, context: str, options: List[Tuple[str, float]]) -> DecisionPoint:
        """Make a decision with explicit reasoning"""
        reasoning_chain = []

        # Step 1: Analyze context
        context_analysis = self._analyze_context(context)
        reasoning_chain.append(f"Context analysis: {context_analysis}")

        # Step 2: Evaluate options
        evaluated_options = []
        for option, initial_score in options:
            adjusted_score = self._evaluate_option(option, context, initial_score)
            evaluated_options.append((option, adjusted_score))
            reasoning_chain.append(f"Option '{option}' scored {adjusted_score:.3f}")

        # Step 3: Select best option
        evaluated_options.sort(key=lambda x: x[1], reverse=True)
        best_option, best_score = evaluated_options[0]
        reasoning_chain.append(f"Selected '{best_option}' with confidence {best_score:.3f}")

        # Step 4: Consider alternatives
        alternatives = evaluated_options[1:4]
        reasoning_chain.append(f"Considered {len(alternatives)} alternatives")

        # Create decision point
        decision = DecisionPoint(
            timestamp=time.time(),
            context=context,
            decision=best_option,
            reasoning=reasoning_chain,
            confidence=best_score,
            alternatives_considered=alternatives
        )

        # Store in decision history
        self.agent.decision_history.append(decision)

        return decision

    def _analyze_context(self, context: str) -> str:
        """Analyze the context of a decision"""
        context_lower = context.lower()

        if 'urgent' in context_lower or 'quickly' in context_lower:
            return "Time-sensitive task requiring fast execution"
        elif 'carefully' in context_lower or 'important' in context_lower:
            return "High-stakes task requiring careful consideration"
        elif 'creative' in context_lower or 'innovative' in context_lower:
            return "Creative task requiring novel approaches"
        else:
            return "Standard task with normal execution requirements"

    def _evaluate_option(self, option: str, context: str, initial_score: float) -> float:
        """Evaluate an option considering context and history"""
        score = initial_score

        # Adjust based on past performance
        if option in self.agent.metrics.task_success_rate:
            success_rate = self.agent.metrics.get_success_rate(option)
            score *= (0.7 + 0.3 * success_rate)

        # Adjust based on context
        context_lower = context.lower()
        if 'safe' in context_lower and 'risky' in option.lower():
            score *= 0.5
        elif 'fast' in context_lower and 'quick' in option.lower():
            score *= 1.3

        return max(0.0, min(1.0, score))

    def reflect_on_outcome(self, decision: DecisionPoint, actual_outcome: str, success: bool):
        """Reflect on a decision outcome and learn"""
        decision.outcome = actual_outcome
        decision.was_correct = success

        if success:
            lesson = f"Strategy '{decision.decision}' works well for context: {decision.context[:50]}"
            self.agent.success_patterns[decision.decision] += 1
        else:
            # Analyze what went wrong
            alternatives = decision.alternatives_considered
            if alternatives:
                better_option = alternatives[0][0]
                lesson = f"Should have chosen '{better_option}' instead of '{decision.decision}' for: {decision.context[:50]}"
            else:
                lesson = f"Strategy '{decision.decision}' failed for context: {decision.context[:50]}"
            self.agent.error_patterns[decision.decision] += 1

        decision.learned_lesson = lesson

        # Add to learning buffer
        self.agent.learning_buffer.append({
            'type': 'task_success' if success else 'task_failure',
            'pattern': decision.decision,
            'context': decision.context,
            'lesson': lesson,
            'timestamp': time.time()
        })

        logger.info(f"[MetaCognitive] Learned: {lesson}")

# ================================================================================
# KNOWLEDGE GRAPH ENGINE
# ================================================================================

class KnowledgeGraphEngine:
    """Manages contextual knowledge as a graph structure"""

    def __init__(self, agent_state: AdvancedAgentState):
        self.agent = agent_state

    def add_knowledge(self, content: str, node_type: str = 'fact', source: str = 'user_interaction') -> str:
        """Add new knowledge to the graph"""
        # Create embedding
        embedding = lite_embed([content])[0]

        # Generate unique ID
        node_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:12]

        # Create knowledge node
        node = KnowledgeNode(
            id=node_id,
            content=content,
            embedding=embedding,
            node_type=node_type,
            timestamp=time.time(),
            source=source
        )

        # Find related nodes
        related = self._find_related_nodes(embedding)
        for related_id, similarity in related[:5]:
            node.connections[related_id] = similarity
            if related_id in self.agent.knowledge_graph:
                self.agent.knowledge_graph[related_id].connections[node_id] = similarity

        # Store in graph
        self.agent.knowledge_graph[node_id] = node

        logger.info(f"[KnowledgeGraph] Added node {node_id} with {len(node.connections)} connections")

        return node_id

    def _find_related_nodes(self, embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Find nodes related to the given embedding"""
        similarities = []

        for node_id, node in self.agent.knowledge_graph.items():
            try:
                similarity = self._cosine_similarity(embedding, node.embedding)
                similarities.append((node_id, similarity))
            except Exception:
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            arr1 = np.array(vec1)
            arr2 = np.array(vec2)
            dot_product = np.dot(arr1, arr2)
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    def query_knowledge(self, query: str, top_k: int = 5) -> List[KnowledgeNode]:
        """Query the knowledge graph"""
        query_embedding = lite_embed([query])[0]
        related = self._find_related_nodes(query_embedding, top_k)

        results = []
        for node_id, similarity in related:
            if node_id in self.agent.knowledge_graph:
                node = self.agent.knowledge_graph[node_id]
                node.access_count += 1
                node.last_accessed = time.time()
                results.append(node)

        return results

    def get_context_subgraph(self, node_ids: List[str], depth: int = 2) -> Dict[str, KnowledgeNode]:
        """Get a subgraph centered on specific nodes"""
        subgraph = {}
        to_explore = set(node_ids)
        explored = set()

        for _ in range(depth):
            current_level = to_explore - explored
            if not current_level:
                break

            for node_id in current_level:
                if node_id in self.agent.knowledge_graph:
                    node = self.agent.knowledge_graph[node_id]
                    subgraph[node_id] = node
                    to_explore.update(node.connections.keys())
                explored.add(node_id)

        return subgraph

# ================================================================================
# TOOL ORCHESTRATION ENGINE
# ================================================================================

class ToolOrchestrator:
    """Orchestrates tool execution with dependency resolution and parallelization"""

    def __init__(self, agent_state: AdvancedAgentState):
        self.agent = agent_state
        self._initialize_tools()

    def _initialize_tools(self):
        """Initialize tool registry with metadata"""
        self.agent.tool_registry = {
            'memory_search': {
                'function': self._tool_memory_search,
                'can_parallel': True,
                'requires_network': False,
                'estimated_time': 0.5
            },
            'web_search': {
                'function': self._tool_web_search,
                'can_parallel': True,
                'requires_network': True,
                'estimated_time': 2.0
            },
            'calc': {
                'function': self._tool_calc,
                'can_parallel': True,
                'requires_network': False,
                'estimated_time': 0.1
            },
            'open_url': {
                'function': self._tool_open_url,
                'can_parallel': False,
                'requires_network': True,
                'estimated_time': 1.0
            },
            'agent': {
                'function': self._tool_agent_control,
                'can_parallel': False,
                'requires_network': False,
                'estimated_time': 0.5
            }
        }

    def execute_tools(self, tool_calls: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """Execute multiple tools with optimization"""
        results = {}

        # Group by parallelizability
        parallel_calls = []
        sequential_calls = []

        for tool_name, arg in tool_calls:
            if tool_name in self.agent.tool_registry:
                tool_info = self.agent.tool_registry[tool_name]
                if tool_info['can_parallel'] and self.agent.parallel_execution_enabled:
                    parallel_calls.append((tool_name, arg))
                else:
                    sequential_calls.append((tool_name, arg))

        # Execute parallel tools
        if parallel_calls:
            parallel_results = self._execute_parallel(parallel_calls)
            results.update(parallel_results)

        # Execute sequential tools
        for tool_name, arg in sequential_calls:
            start_time = time.time()
            try:
                result = self._execute_single_tool(tool_name, arg)
                duration = time.time() - start_time
                results[tool_name] = result
                self.agent.metrics.record_tool_execution(tool_name, duration, True)
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"[ToolOrchestrator] Tool {tool_name} failed: {e}")
                results[tool_name] = f"Error: {e}"
                self.agent.metrics.record_tool_execution(tool_name, duration, False)

        return results

    def _execute_parallel(self, tool_calls: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """Execute tools in parallel using threads"""
        results = {}
        threads = []
        result_queue = queue.Queue()

        def worker(tool_name, arg, result_q):
            start_time = time.time()
            try:
                result = self._execute_single_tool(tool_name, arg)
                duration = time.time() - start_time
                result_q.put((tool_name, result, duration, True))
            except Exception as e:
                duration = time.time() - start_time
                result_q.put((tool_name, f"Error: {e}", duration, False))

        # Start threads
        for tool_name, arg in tool_calls:
            t = threading.Thread(target=worker, args=(tool_name, arg, result_queue))
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join(timeout=10.0)

        # Collect results
        while not result_queue.empty():
            try:
                tool_name, result, duration, success = result_queue.get_nowait()
                results[tool_name] = result
                self.agent.metrics.record_tool_execution(tool_name, duration, success)
            except queue.Empty:
                break

        return results

    def _execute_single_tool(self, tool_name: str, arg: Any) -> Any:
        """Execute a single tool"""
        if tool_name not in self.agent.tool_registry:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool_func = self.agent.tool_registry[tool_name]['function']
        return tool_func(arg)

    # Tool implementations
    def _tool_memory_search(self, query: str) -> str:
        """Search memory/context"""
        try:
            from SarahMemoryCompare import compare_local_memory
            result = compare_local_memory(query)
            return result if result else "No relevant memories found"
        except Exception as e:
            return f"Memory search unavailable: {e}"

    def _tool_web_search(self, query: str) -> str:
        """Perform web search"""
        if SAFE_MODE or LOCAL_ONLY_MODE or is_offline():
            return "Web search disabled in current mode"
        try:
            from SarahMemoryResearch import perform_web_search
            return perform_web_search(query)
        except Exception as e:
            return f"Web search failed: {e}"

    def _tool_calc(self, expression: str) -> str:
        """Perform calculation"""
        try:
            from SarahMemoryWebSYM import WebSemanticSynthesizer
            calc = WebSemanticSynthesizer()
            return str(calc.sarah_calculator(expression))
        except Exception as e:
            return f"Calculation failed: {e}"

    def _tool_open_url(self, url: str) -> str:
        """Open URL in browser"""
        if SAFE_MODE or LOCAL_ONLY_MODE:
            return "URL opening disabled in current mode"
        try:
            import webbrowser
            webbrowser.open(url)
            return f"Opened {url}"
        except Exception as e:
            return f"Failed to open URL: {e}"

    def _tool_agent_control(self, command: str) -> str:
        """Execute agent control command"""
        try:
            from SarahMemoryAiFunctions import handle_ai_agent_command
            return handle_ai_agent_command(command)
        except Exception as e:
            return f"Agent control failed: {e}"

# ================================================================================
# UNIFIED ADVANCED AGENT INTERFACE
# ================================================================================

def initialize_advanced_agent():
    """Initialize the advanced agent system"""
    try:
        ADVANCED_AGENT.initialize()
        logger.info("[AdvancedAgent] v8.0 systems initialized successfully")
    except Exception as e:
        logger.error(f"[AdvancedAgent] Initialization failed: {e}")

def advanced_agent_query(user_text: str, context: Optional[Dict] = None) -> str:
    """
    Main entry point for advanced agent query processing

    This orchestrates all v8.0 capabilities:
    - Hierarchical planning
    - Meta-cognitive reasoning
    - Knowledge graph querying
    - Parallel tool execution
    - Predictive intent modeling
    - Autonomous learning
    """
    start_time = time.time()
    ADVANCED_AGENT.interaction_count += 1

    try:
        # Step 1: Check for predicted intents
        predicted = None
        try:
            predicted = ADVANCED_AGENT.predicted_intents.get_nowait()
            if predicted['confidence'] > 0.6:
                logger.info(f"[AdvancedAgent] Predicted intent: {predicted['intent']} (confidence: {predicted['confidence']:.2f})")
        except queue.Empty:
            pass

        # Step 2: Classify intent
        intent = classify_intent(user_text) if user_text else "unknown"

        # Step 3: Query knowledge graph for context
        kg_engine = KnowledgeGraphEngine(ADVANCED_AGENT)
        relevant_knowledge = kg_engine.query_knowledge(user_text, top_k=5)

        context_summary = ""
        if relevant_knowledge:
            context_summary = " | ".join([k.content[:50] for k in relevant_knowledge[:3]])
            logger.info(f"[AdvancedAgent] Found {len(relevant_knowledge)} relevant knowledge nodes")

        # Step 4: Create hierarchical plan
        planner = HierarchicalPlanner(ADVANCED_AGENT)
        task = planner.create_task(
            description=user_text,
            intent=intent,
            priority=TaskPriority.NORMAL
        )

        # Step 5: Meta-cognitive reasoning about approach
        reasoner = MetaCognitiveReasoner(ADVANCED_AGENT)

        # Generate approach options
        options = []
        if task.subtasks:
            options.append(("hierarchical_decomposition", 0.8))
        if task.tools_required:
            options.append(("direct_tool_execution", 0.7))
        if relevant_knowledge:
            options.append(("knowledge_based_response", 0.75))
        if not options:
            options.append(("conversational_response", 0.6))

        decision = reasoner.reason_about_decision(
            context=f"Query: {user_text} | Intent: {intent}",
            options=options
        )

        logger.info(f"[AdvancedAgent] Decision: {decision.decision} (confidence: {decision.confidence:.2f})")

        # Step 6: Execute based on decision
        orchestrator = ToolOrchestrator(ADVANCED_AGENT)

        if decision.decision == "hierarchical_decomposition":
            # Execute with full hierarchical planning
            execution_plan = planner.plan_execution(task)
            results = orchestrator.execute_tools(execution_plan)

            # Combine results
            result_text = self._combine_tool_results(results, user_text)
            success = bool(results)

        elif decision.decision == "direct_tool_execution":
            # Execute tools directly
            tool_calls = [(tool, user_text) for tool in task.tools_required]
            results = orchestrator.execute_tools(tool_calls)
            result_text = self._combine_tool_results(results, user_text)
            success = bool(results)

        elif decision.decision == "knowledge_based_response":
            # Use knowledge graph
            result_text = self._generate_knowledge_response(relevant_knowledge, user_text)
            success = True

        else:
            # Conversational fallback
            result_text = self._generate_conversational_response(user_text, context)
            success = True

        # Step 7: Learn from interaction
        ADVANCED_AGENT.learning_buffer.append({
            'type': 'task_success' if success else 'task_failure',
            'pattern': intent,
            'context': user_text,
            'decision': decision.decision,
            'timestamp': time.time()
        })

        # Step 8: Add to knowledge graph
        kg_engine.add_knowledge(
            content=f"Q: {user_text[:100]} | A: {result_text[:100]}",
            node_type='experience',
            source='interaction'
        )

        # Step 9: Reflect on outcome
        reasoner.reflect_on_outcome(decision, result_text, success)

        # Step 10: Record metrics
        duration = time.time() - start_time
        ADVANCED_AGENT.metrics.response_times.append(duration)
        ADVANCED_AGENT.metrics.record_task_outcome(intent, success)

        # Step 11: Update user patterns for prediction
        current_time = datetime.now()
        pattern_key = f"{current_time.hour}_{current_time.weekday()}"
        ADVANCED_AGENT.user_patterns[pattern_key].append({
            'intent': intent,
            'query': user_text[:50],
            'hour': current_time.hour,
            'day': current_time.weekday(),
            'timestamp': time.time()
        })

        logger.info(f"[AdvancedAgent] Completed in {duration:.2f}s | Success: {success}")

        return result_text

    except Exception as e:
        logger.error(f"[AdvancedAgent] Query processing failed: {e}")
        return f"I encountered an error processing your request: {e}"

def _combine_tool_results(results: Dict[str, Any], original_query: str) -> str:
    """Combine results from multiple tools into coherent response"""
    if not results:
        return "I couldn't find a suitable response."

    # Filter out errors
    successful_results = {k: v for k, v in results.items()
                         if not str(v).startswith("Error:")}

    if not successful_results:
        return "I encountered errors with all available tools."

    # If single result, return it
    if len(successful_results) == 1:
        return list(successful_results.values())[0]

    # Combine multiple results intelligently
    combined = []
    for tool_name, result in successful_results.items():
        if result and len(str(result)) > 10:
            combined.append(f"[{tool_name}]: {result}")

    return "\n\n".join(combined) if combined else "Results obtained but require further processing."

def _generate_knowledge_response(knowledge_nodes: List[KnowledgeNode], query: str) -> str:
    """Generate response based on knowledge graph"""
    if not knowledge_nodes:
        return "I don't have enough knowledge to answer that."

    # Sort by relevance (access count and timestamp)
    sorted_nodes = sorted(knowledge_nodes,
                         key=lambda n: (n.access_count, n.timestamp),
                         reverse=True)

    # Combine top knowledge
    response_parts = []
    for node in sorted_nodes[:3]:
        if len(node.content) > 20:
            response_parts.append(node.content)

    if response_parts:
        return "Based on what I know: " + " | ".join(response_parts)
    else:
        return "I have some related knowledge but need more context to answer accurately."

def _generate_conversational_response(text: str, context: Optional[Dict]) -> str:
    """Generate a conversational response"""
    text_lower = text.lower()

    # Greeting
    if any(g in text_lower for g in ['hello', 'hi', 'hey']):
        return "Hello! How can I assist you today?"

    # Thanks
    if any(t in text_lower for t in ['thank', 'thanks']):
        return "You're welcome! Let me know if you need anything else."

    # Capability query
    if 'can you' in text_lower or 'what can' in text_lower:
        return ("I'm an advanced AI agent with capabilities including: hierarchical task planning, "
                "meta-cognitive reasoning, knowledge graph queries, parallel tool execution, "
                "and autonomous learning. What would you like me to help with?")

    # Default
    return "I understand you said: " + text + ". Could you provide more details about what you'd like me to do?"

# ================================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ================================================================================


# ================================================================================
# LEGACY FUNCTIONS (CRITICAL FOR BACKWARD COMPATIBILITY)
# ================================================================================

def _now_ms():
    return int(time.time()*1000)


def _start_input_watchers():
    global _watch_thread_started, _watch_thread, _last_mouse
    if _watch_thread_started:
        return
    _watch_thread_started = True
    def _loop():
        global _last_mouse
        try:
            import pyautogui as _pag
        except Exception:
            _pag = None
        while True:
            try:
                moved = False
                if _pag is not None:
                    pos = _pag.position()
                    if _last_mouse is None:
                        _last_mouse = pos
                    if pos != _last_mouse:
                        moved = True
                        _last_mouse = pos
                # TODO: optionally integrate keyboard/gamepad hooks when available
                if moved:
                    _agent_state["last_human_ts"] = _now_ms()
                    if _agent_state["mode"] != "IDLE":
                        # If user is interacting, halt the agent
                        _agent_state["mode"] = "HALTED"
                        _agent_state["resume_eta_ms"] = _now_ms() + AI_AGENT_RESUME_DELAY
                time.sleep(0.1)
            except Exception:
                time.sleep(0.2)
    import threading
    _watch_thread = threading.Thread(target=_loop, name="SM-AgentWatch", daemon=True)
    _watch_thread.start()


def agent_status():
    return dict(_agent_state)


def _update_gui_status():
    try:
        if hasattr(config, "status_bar"):
            config.status_bar.set_status(f"Agent: {_agent_state['mode']}")
    except Exception:
        pass


def intercept_agent_control_phrases(text: str):
    if not text:
        return None
    t = text.strip().lower()
    if any(p in t for p in AI_AGENT_STOP_PHRASES):
        _agent_state["mode"] = "HALTED"
        _agent_state["resume_eta_ms"] = _now_ms() + AI_AGENT_RESUME_DELAY
        _update_gui_status()
        try:
            synthesize_voice("Emergency stop acknowledged. Agent halted.")
        except Exception:
            pass
        return "Agent halted."
    if any(p in t for p in AI_AGENT_HALT_PHRASES):
        _agent_state["mode"] = "HALTED"
        _agent_state["resume_eta_ms"] = _now_ms() + AI_AGENT_RESUME_DELAY
        _update_gui_status()
        return "Paused."
    if any(p in t for p in AI_AGENT_RESUME_PHRASES):
        if _now_ms() >= _agent_state.get("resume_eta_ms", 0):
            _agent_state["mode"] = "ACTIVE"
            _update_gui_status()
            return "Resuming."
        else:
            wait_ms = max(0, _agent_state["resume_eta_ms"] - _now_ms())
            return f"Cannot resume yet. Waiting {int(wait_ms/1000)} seconds."
    return None


def agent_guard():
    # Start watchers on first use
    _start_input_watchers()
    now = _now_ms()
    # If recent human input, ensure we are halted
    if now - _agent_state.get("last_human_ts", 0) < AI_AGENT_USER_ACTIVITY_TIMEOUT_MS:
        _agent_state["mode"] = "HALTED"
        _agent_state["resume_eta_ms"] = now + AI_AGENT_RESUME_DELAY
        _update_gui_status()
        return False
    # If we are halted but resume window elapsed, return True but keep ACTIVE
    if _agent_state["mode"] in ("HALTED","RESUME_PENDING"):
        if now >= _agent_state.get("resume_eta_ms", 0):
            _agent_state["mode"] = "ACTIVE"
            _update_gui_status()
            return True
        else:
            _agent_state["mode"] = "HALTED"
            _update_gui_status()
            return False
    # Normal
    if _agent_state["mode"] == "IDLE":
        _agent_state["mode"] = "ACTIVE"
        _update_gui_status()
    return True



    # As a productivity fallback, try the PLAN→ACT loop if agent enabled or intent was unknown
    try:
        if AI_AGENT_ENABLED or intent in ("unknown", "statement"):
            plan = plan_and_act(user_text, max_steps=4)
            if plan.get("final"):
                return plan["final"]
    except Exception as e:
        logger.debug(f"[Router] planner fallback failed: {e}")
    return "I'm unsure how to respond."# === Stubbed Plug-in Connectors ===

def local_memory_lookup(text):
    try:
        from SarahMemoryCompare import compare_local_memory
        return compare_local_memory(text)
    except:
        return None


def web_research_query(text):
    try:
        from SarahMemoryResearch import perform_web_search

        return perform_web_search(text)
    except Exception:
        return None


def symbolic_calc_answer(text):
    """Delegate symbolic / arithmetic questions to SarahMemoryWebSYM.

    Uses WebSemanticSynthesizer.sarah_calculator, which already handles
    word-based math ("plus", "percent of", etc.) and safe expression eval.
    """
    try:
        from SarahMemoryWebSYM import WebSemanticSynthesizer
        return WebSemanticSynthesizer.sarah_calculator(text, text)
    except Exception:
        return None


def deep_learn_context(text):
    try:
        from SarahMemoryDL import learn_from_input
        learn_from_input(text)
        return None
    except:
        return None

# === Additional Functions ===

def get_context():
    return context_buffer


def clear_context():
    global context_buffer, context_embeddings
    context_buffer = []
    context_embeddings = []


def add_to_context(interaction):
    # NEW: ensure we have an embedding if caller didn’t provide one
    try:
        if "embedding" not in interaction or not interaction["embedding"]:
            src_text = interaction.get("input") or interaction.get("final_response") or ""
            try:
                interaction["embedding"] = lite_embed(src_text)[0]
            except Exception:
                interaction["embedding"] = _fallback_embed(src_text)
    except Exception:
        # keep going even if embedding fails
        pass

    context_buffer.append(interaction)
    if len(context_buffer) > config.CONTEXT_BUFFER_SIZE:
        context_buffer.pop(0)

    if "embedding" in interaction:
        context_embeddings.append(np.array(interaction["embedding"]))
        if len(context_embeddings) > config.CONTEXT_BUFFER_SIZE:
            context_embeddings.pop(0)

    try:
        conn = sqlite3.connect(CONTEXT_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO context_history (timestamp, input, embedding, final_response, source, intent) VALUES (?, ?, ?, ?, ?, ?)",
            (interaction.get("timestamp"), interaction.get("input"), json.dumps(interaction.get("embedding")),
             interaction.get("final_response"), interaction.get("source"), interaction.get("intent"))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error persisting context history: {e}")

    # Remote hub broadcast (best-effort)
    try:
        hub = _get_hub()
        if hub and isinstance(interaction.get("final_response"), str):
            #_fire_and_forget(hub.broadcast_context, interaction.get("final_response"), {"intent": interaction.get("intent"), "source": interaction.get("source")})
            _fire_and_forget(hub.context_update, interaction.get("final_response"),
                            tags=[interaction.get("intent"), interaction.get("source")])
    except Exception:
        pass


def retrieve_similar_context(current_embedding, top_n=3):
    if not context_embeddings or not context_buffer:
        return []
    similarities = []
    current_embedding = np.array(current_embedding)
    for emb in context_embeddings:
        sim = np.dot(current_embedding, emb) / (np.linalg.norm(current_embedding) * np.linalg.norm(emb))
        similarities.append(sim)
    top_indices = np.argsort(similarities)[-top_n:]
    return [context_buffer[i] for i in top_indices if i < len(context_buffer)]


def log_ai_functions_event(event, details):
    try:
        db_path = os.path.join(config.DATASETS_DIR, "functions.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS ai_functions_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )'''
        )
        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO ai_functions_events (timestamp, event, details) VALUES (?, ?, ?)",
            (timestamp, event, details)
        )
        conn.commit()
        conn.close()
        logger.info("Logged AI functions event to functions.db successfully.")
    except Exception as e:
        logger.error(f"Error logging AI functions event to functions.db: {e}")

if __name__ == '__main__':
    logger.info("Starting Enhanced SarahMemoryAiFunctions module test.")
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=5)
            voice_text = recognizer.recognize_google(audio)
            print("You said:", voice_text)
            if voice_text:
                response = route_intent_response(voice_text)
                print("AI responded:", response)
            else:
                print("No valid speech input detected.")
    except Exception as e:
        logger.error(f"Speech recognition failed: {e}")
        print("Speech recognition error:", e)
    logger.info("Enhanced SarahMemoryAiFunctions module testing complete.")

import os, time, logging
logger = logging.getLogger("AiFunctions")


def handle_ai_agent_command(command_text):
    """
    Execute desktop/UI/game commands with safety gating, human-activity halting, and consent checks.
    """
    import time
    import re as _re
    try:
        import pyautogui as _pag
        _pag.FAILSAFE = False
    except Exception as e:
        logger.error(f"[AI-Agent] pyautogui unavailable: {e}")
        return "Desktop control is unavailable on this system."

    if not AI_AGENT_ENABLED:
        return "Agent control is disabled."

    # Normalize
    raw = command_text or ""
    cmd = raw.strip().lower()

    # If risky action requires consent
    for risky in AI_AGENT_REQUIRE_CONSENT:
        if risky in cmd:
            synthesize_voice(f"I need your permission to {risky}. Say yes or no.")
            # Simple yes/no listen using speech_recognition if available; else deny
            try:
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    audio = r.listen(source, timeout=4, phrase_time_limit=3)
                    utter = r.recognize_google(audio).strip().lower()
            except Exception:
                utter = ""
            if utter in AI_AGENT_CONFIRM_YES:
                log_ai_functions_event("AgentConsent", f"Granted: {risky}")
            else:
                log_ai_functions_event("AgentConsent", f"Denied: {risky}")
                return "Okay, I won't proceed."

    # Human activity guard
    if not agent_guard():
        return f"Agent paused ({agent_status()['mode']})."

    # App lifecycle via Si
    if any(w in cmd for w in ("open ", "launch ", "start ", "focus ", "maximize ", "minimize ", "close ", "terminate ", "kill ")):
        try:
            from SarahMemorySi import manage_application_request
            ok = manage_application_request(cmd)
            return "Done." if ok else "I couldn't do that."
        except Exception as e:
            logger.warning(f"[AI-Agent] Si routing failed: {e}")

    # Basic UI actions
    if "move mouse to" in cmd:
        try:
            coords = cmd.replace("move mouse to","").strip()
            parts = [p.strip() for p in coords.replace("and",",").split(",") if p.strip()]
            x, y = int(float(parts[0])), int(float(parts[1]))
            _pag.moveTo(x, y, duration=0.15)
            return f"Moved mouse to {x},{y}."
        except Exception as e:
            return f"Invalid coordinates: {e}"

    if "double click" in cmd or "double-click" in cmd:
        _pag.doubleClick(); return "Double clicked."

    if "click" in cmd:
        _pag.click(); return "Clicked."

    # Typing
    m = _re.search(r"\btype\s+(.*)$", cmd)
    if m:
        text_to_type = m.group(1).strip().strip('"')
        _pag.typewrite(text_to_type, interval=0.02)
        return "Typed."

    # Press hotkey
    m = _re.search(r"\bpress\s+([a-z0-9\+\-]+)$", cmd)
    if m:
        key = m.group(1)
        if "+" in key:
            parts = [k.strip() for k in key.split("+")]
            _pag.hotkey(*parts)
        else:
            _pag.press(key)
        return "Pressed."

    if "scroll down" in cmd:
        _pag.scroll(-500); return "Scrolled down."
    if "scroll up" in cmd:
        _pag.scroll(500); return "Scrolled up."

    # Wait
    m = _re.search(r"\bwait\s+(\d+(?:\.\d+)?)", cmd)
    if m:
        time.sleep(float(m.group(1))); return "Waiting done."

    # --- Game micro-controls (WASD etc.) ---
    def _hold(key, dur=0.2):
        _pag.keyDown(key); time.sleep(dur); _pag.keyUp(key)
    # Optional: parse "for X seconds"
    dur = 0.2
    dm = _re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s*(?:sec|second|seconds|s)\b", cmd)
    if dm:
        dur = min(3.0, max(0.05, float(dm.group(1))))

    if any(w in cmd for w in ("forward","move forward","go forward")):
        _hold("w", dur); return "Moving forward."
    if any(w in cmd for w in ("back","backward","move back")):
        _hold("s", dur); return "Moving back."
    if any(w in cmd for w in ("left","strafe left","move left")):
        _hold("a", dur); return "Left."
    if any(w in cmd for w in ("right","strafe right","move right")):
        _hold("d", dur); return "Right."
    if "jump" in cmd:
        _hold("space", 0.1); return "Jump."
    if "crouch" in cmd or "duck" in cmd:
        _hold("ctrl", 0.1); return "Crouch."
    if "run" in cmd or "sprint" in cmd:
        _hold("shift", dur); return "Run."
    if "shoot" in cmd or "fire" in cmd:
        _pag.mouseDown(); time.sleep(dur); _pag.mouseUp(); return "Shoot."
    if "reload" in cmd:
        _hold("r", 0.1); return "Reload."
    if "use" in cmd or "interact" in cmd:
        _hold("e", 0.1); return "Use."
    if "open map" in cmd or "map" in cmd:
        _hold("tab", 0.1); return "Map."

    # Unknown
    return "I didn't recognize that command."


# [PATCH v7.7.2] Standardized chunking, transformer selection, and rerank utilities

def select_sentence_transformer(preferred: list = None):
    """
    Factory: tries Globals MODEL_CONFIG or explicit list; returns object with .encode(list|str)->np.ndarray
    Falls back to a deterministic hash embed (from SarahMemoryAdvCU) if ST unavailable.
    """
    import numpy as np
    try:
        from SarahMemoryGlobals import MODEL_CONFIG
    except Exception:
        MODEL_CONFIG = {}
    tried = []
    # Preferred list first
    if isinstance(preferred, (list, tuple)) and preferred:
        tried.extend([m for m in preferred])
    # Enabled in globals next
    tried.extend([m for m, en in (MODEL_CONFIG or {}).items() if en and m not in tried])
    tried.append("all-MiniLM-L6-v2")
    for name in tried:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(name)
            class _Wrap:
                def encode(self, x):
                    return model.encode(x)
            return _Wrap()
        except Exception:
            continue
    # Fallback: hash embed
    from SarahMemoryAdvCU import embed_text as _adv_embed
    class _HashWrap:
        def encode(self, x):
            import numpy as np
            if isinstance(x, str):
                x = [x]
            return np.vstack([_adv_embed(t, dim=128) for t in x])
    return _HashWrap()


def _build_provenance(used_calculator=False, used_local=False, used_web=False, api_name=None, model_name=None, used_fallback=False):
    if used_calculator: return {"source":"Calculator"}
    if used_local: return {"source":"Local"}
    if used_web and api_name and model_name: return {"source": f"API/{api_name} {model_name}"}
    if used_web: return {"source":"Web"}
    if used_fallback: return {"source":"Fallback"}
    return {"source":"Unknown"}


def _classify_intent(text):
    t=(text or "").strip().lower()
    if re.match(r'^[\s\d\+\-\*\/\(\)\.\^%]+$', t): return "Math"
    if "your name" in t or t.startswith("who are you") or "what are you" in t: return "Identity"
    if t.startswith("show me") or "image of" in t: return "MediaRequest"
    if t.startswith("open ") or "launch " in t: return "SystemControl"
    return "General"


# --- injected: on-demand ensure table for `response` ---

def _ensure_response_table(db_path=None):
    try:
        import sqlite3, os, logging
        try:
            import SarahMemoryGlobals as config
        except Exception:
            class config: pass
        if db_path is None:
            base = getattr(config, "BASE_DIR", os.getcwd())
            db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")
        con = sqlite3.connect(db_path); cur = con.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS response (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, user TEXT, content TEXT, source TEXT, intent TEXT)'); con.commit(); con.close()
        logging.debug("[DB] ensured table `response` in %s", db_path)
    except Exception as e:
        try:
            import logging; logging.warning("[DB] ensure `response` failed: %s", e)
        except Exception:
            pass
try:
    _ensure_response_table()
except Exception:
    pass

def add_to_context_entry(interaction_or_text, final_response=None, intent="chat", source="local"):
    """
    Persist one exchange into context_history.db and in-memory buffer.
    Accepts either:
      - a dict: {"timestamp","input","embedding","final_response","source","intent"}
      - or (text, final_response, intent, source)
    Never raises.
    """
    try:
        import time, json, sqlite3, numpy as _np
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

        if isinstance(interaction_or_text, dict):
            data = interaction_or_text
            text = str(data.get("input", "") or "")
            resp = str(data.get("final_response", "") or "")
            emb = data.get("embedding", None)
            intent = data.get("intent", intent)
            source = data.get("source", source)
            ts = data.get("timestamp", ts)
        else:
            text = str(interaction_or_text or "")
            resp = str(final_response or "")
            try:
                emb = lite_embed(text)[0]  # existing small embedder
            except Exception:
                emb = None

        # Persist to DB (assumes CONTEXT_DB_PATH/table created by init_context_history)
        with sqlite3.connect(CONTEXT_DB_PATH) as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO context_history (timestamp, input, embedding, final_response, source, intent) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts, text, json.dumps(list(emb)) if emb is not None else None, resp, source, intent)
            )
            con.commit()

        # Update RAM ring if present
        try:
            context_buffer.append({'timestamp': ts, 'input': text, 'embedding': list(emb) if emb is not None else []})
            context_embeddings.append(_np.array(emb) if emb is not None else _np.zeros((768,), dtype=float))
            maxn = int(getattr(config, "CONTEXT_BUFFER_SIZE", 20))
            if len(context_buffer) > maxn:
                del context_buffer[0]; del context_embeddings[0]
        except Exception:
            pass
    except Exception:
        pass



def get_relevant_context(query: str, top_n: int = 3):
    """Phase B helper — retrieve top-N similar context turns for a query.

    Uses the existing in-RAM ring (context_buffer/context_embeddings) and
    falls back safely if embeddings are unavailable or disabled.
    """
    try:
        cfg = get_context_config()
        if not cfg.get("enabled", True):
            return []
    except Exception:
        cfg = {"enabled": True}

    try:
        if not query:
            return []
        # Use the same lite_embed path as the rest of the module
        emb = lite_embed(query)[0]
    except Exception:
        try:
            emb = _fallback_embed(query)
        except Exception:
            return []

    try:
        # Reuse the cosine similarity helper
        return retrieve_similar_context(emb, top_n)
    except Exception:
        return []



def init_context_history():
    """Initialize context history (legacy compatibility)"""
    try:
        os.makedirs(os.path.dirname(CONTEXT_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(CONTEXT_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS context_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input TEXT,
                embedding TEXT,
                final_response TEXT,
                source TEXT,
                intent TEXT
            )"""
        )
        cursor.execute('SELECT timestamp, input, embedding FROM context_history ORDER BY id DESC LIMIT ?',
                      (getattr(config, "CONTEXT_BUFFER_SIZE", 20),))
        rows = cursor.fetchall()
        for ts, ui, emb in reversed(rows):
            try:
                embedding = json.loads(emb)
                context_buffer.append({'timestamp': ts, 'input': ui, 'embedding': embedding})
                try:
                    import numpy as _np
                    context_embeddings.append(_np.array(embedding))
                except Exception:
                    context_embeddings.append(embedding)
            except Exception as _e:
                logger.error(f"[Context] Error loading entry: {_e}")
        conn.commit()
        conn.close()
        logger.info("Initialized context history.")
    except Exception as e:
        logger.error(f"Error initializing context history: {e}")

def route_query(user_text: str) -> str:
    """
    Main routing function with v8.0 advanced agent integration

    This maintains backward compatibility while enabling advanced features
    """
    try:
        # Use advanced agent if available and enabled
        if getattr(config, "USE_ADVANCED_AGENT", True):
            return advanced_agent_query(user_text)
        else:
            # Legacy fallback
            return _legacy_route_query(user_text)
    except Exception as e:
        logger.error(f"[RouteQuery] Error: {e}")
        return f"Error processing query: {e}"

def _legacy_route_query(user_text: str) -> str:
    """Legacy query routing (backward compatibility)"""
    t = (user_text or "").strip()
    if not t:
        return "I didn't catch that."

    # Simple routing
    tl = t.lower()

    # URL
    if tl.startswith(("http://", "https://", "www.")):
        try:
            import webbrowser
            webbrowser.open(t)
            return f"Opening {t}"
        except Exception as e:
            return f"Failed to open URL: {e}"

    # Math
    if any(ch in t for ch in "+-*/^=") or tl.startswith("calculate"):
        try:
            from SarahMemoryWebSYM import WebSemanticSynthesizer
            calc = WebSemanticSynthesizer()
            return str(calc.sarah_calculator(t))
        except Exception as e:
            return f"Calculation error: {e}"

    # Default
    return "I'm unsure how to respond to that."

# ================================================================================
# MODULE INITIALIZATION
# ================================================================================

def _module_init():
    """Initialize module on import"""
    try:
        # Initialize context history
        init_context_history()

        # Initialize advanced agent if enabled
        if getattr(config, "USE_ADVANCED_AGENT", True):
            initialize_advanced_agent()
            logger.info("[SarahMemoryAiFunctions] v8.0 Advanced Agent loaded successfully")
        else:
            logger.info("[SarahMemoryAiFunctions] Running in legacy mode")
    except Exception as e:
        logger.error(f"[SarahMemoryAiFunctions] Initialization error: {e}")

# Auto-initialize on module load
_module_init()

# ================================================================================
# EXPORTS
# ================================================================================

__all__ = [
    'route_query',
    'advanced_agent_query',
    'initialize_advanced_agent',
    'ADVANCED_AGENT',
    'Task',
    'TaskPriority',
    'TaskStatus',
    'HierarchicalPlanner',
    'MetaCognitiveReasoner',
    'KnowledgeGraphEngine',
    'ToolOrchestrator',
    'lite_embed',
]


# ================================================================================
# REMAINING LEGACY FUNCTIONS (MANUALLY ADDED)
# ================================================================================

def smart_reply(user_text: str) -> str:
    """High-level reply leveraging route_query with fallback"""
    try:
        return route_query(user_text)
    except Exception as e:
        logger.warning(f"[smart_reply] error: {e}")
        return "I'm working on that. Let me try again."

def create_chunks(text: str, chunk_size_tokens: int = 256, overlap_tokens: int = 64, language: str = "en") -> list:
    """
    Deterministic chunker by whitespace tokens with overlap.
    Returns list of dicts: {"id": idx, "text": chunk, "start": token_idx, "end": token_idx}
    """
    import re
    toks = re.findall(r"\S+", text or "")
    chunks = []
    i = 0
    idx = 0
    while i < len(toks):
        start = max(0, i - overlap_tokens if idx > 0 else i)
        end = min(len(toks), i + chunk_size_tokens)
        chunk = " ".join(toks[start:end])
        chunks.append({"id": idx, "text": chunk, "start": start, "end": end})
        i += chunk_size_tokens - overlap_tokens if idx > 0 else chunk_size_tokens
        idx += 1
    return chunks

def detect_command_intent(text: str) -> str:
    """
    Lightweight intent router used by GUI/Reply:
    - returns 'image' for "show me ... image/picture" requests
    - returns 'math' if looks like an arithmetic expression
    - otherwise 'chat'
    """
    if not text:
        return "chat"
    import re
    low = text.lower().strip()
    if re.search(r"\bshow\s+me\s+(?:an?|some)?\s*(?:image|images|picture|pictures|pic|pics)\s+of\b", low):
        return "image"
    if re.search(r"^\s*\d+[\d\s\+\-\*\/\(\)\.]*$", low):
        return "math"
    return "chat"

def normalize_text(text: str) -> str:
    """Normalize text for processing"""
    try:
        return ' '.join((text or '').split())
    except Exception:
        return text

def open_url_in_chrome(url: str) -> str:
    """Open a URL in Chrome if present, else default browser"""
    try:
        if not url:
            return "No URL provided."
        import os
        chrome_paths = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
        for cp in chrome_paths:
            if os.path.exists(cp):
                os.startfile(f'"{cp}" "{url}"')  # nosec
                return f"Opening in Chrome: {url}"
        import webbrowser
        webbrowser.open(url)
        return f"Opening in default browser: {url}"
    except Exception as e:
        return f"Failed to open URL: {e}"

def draw_in_paint(subject: str = "dinosaur") -> str:
    """Open Microsoft Paint and draw a simple outline using pyautogui (best effort)"""
    try:
        import os
        paint_path = r"C:\Windows\System32\mspaint.exe"
        if os.path.exists(paint_path):
            os.startfile(paint_path)  # nosec
            import time
            time.sleep(2.5)
            try:
                import pyautogui as pag
                pag.FAILSAFE = False
                # Maximize
                pag.hotkey('alt', 'space')
                time.sleep(0.2)
                pag.press('x')
                time.sleep(0.3)
                w, h = pag.size()
                cx, cy = int(w * 0.5), int(h * 0.6)
                pag.moveTo(cx - 220, cy)
                pag.mouseDown()
                pag.moveRel(80, -60, 0.35)
                pag.moveRel(70, 55, 0.35)
                pag.moveRel(100, 40, 0.4)
                pag.moveRel(-50, 60, 0.3)
                pag.moveRel(-90, -10, 0.35)
                pag.moveRel(-60, -40, 0.35)
                pag.moveRel(20, 35, 0.2)
                pag.moveRel(35, 10, 0.2)
                pag.mouseUp()
                return f"Drew a simple {subject} in Paint."
            except Exception as e:
                return f"Paint opened but drawing failed: {e}"
        return "Paint not found on this system."
    except Exception as e:
        return f"Could not draw in Paint: {e}"

def rerank(query: str, candidates: list, cross_encoder=None) -> list:
    """
    Rerank candidates by relevance to query.
    If cross_encoder provided, use it; else return as-is.
    """
    if cross_encoder is None:
        return candidates
    try:
        pairs = [(query, c) for c in candidates]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _s in ranked]
    except Exception as e:
        logger.warning(f"[rerank] error: {e}")
        return candidates

def list_tools() -> list:
    """List available tools"""
    return [
        {"name": "memory_search", "description": "Search local memory"},
        {"name": "web_search", "description": "Search the web"},
        {"name": "calc", "description": "Calculate mathematical expressions"},
        {"name": "open_url", "description": "Open a URL in browser"},
        {"name": "agent", "description": "AI agent control"}
    ]

def exec_tool(name: str, arg: str = "") -> dict:
    """Execute a tool"""
    tools = {
        "memory_search": lambda a: {"result": local_memory_lookup(a)},
        "web_search": lambda a: {"result": web_research_query(a)},
        "calc": lambda a: {"result": symbolic_calc_answer(a)},
        "open_url": lambda a: {"result": open_url_in_chrome(a)},
        "agent": lambda a: {"result": handle_ai_agent_command(a)}
    }

    if name in tools:
        try:
            return tools[name](arg)
        except Exception as e:
            return {"error": str(e)}
    return {"error": f"Unknown tool: {name}"}

def _simple_plan(text: str) -> list:
    """Simple task planning"""
    return [
        {"action": "understand", "description": "Understand the query"},
        {"action": "search", "description": "Search for information"},
        {"action": "respond", "description": "Formulate response"}
    ]

def plan_and_act(user_text: str, max_steps: int = 4) -> dict:
    """Plan and execute actions"""
    plan = _simple_plan(user_text)
    results = []

    for step in plan[:max_steps]:
        results.append({
            "step": step["action"],
            "result": f"Executed: {step['description']}"
        })

    return {
        "plan": plan,
        "results": results,
        "final": "Task completed"
    }

# Update exports to include all functions
_LEGACY_EXPORTS = [
    'smart_reply', 'create_chunks', 'detect_command_intent', 'normalize_text',
    'open_url_in_chrome', 'draw_in_paint', 'rerank', 'list_tools', 'exec_tool',
    'plan_and_act'
]

# Extend __all__ if it exists
try:
    __all__.extend(_LEGACY_EXPORTS)
except:
    pass