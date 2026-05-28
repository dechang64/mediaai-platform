"""
MediaAI - Cell Culture Media Intelligence Platform
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__license__ = "MIT"

# Core modules
from mediaai.media_vault import MediaVault
from mediaai.media_optimizer import MediaOptimizer
from mediaai.fl_engine import FLEngine
from mediaai.vision_ana import CellSegmentor, ViabilityClassifier
from mediaai.knowledge_hub import KnowledgeHub
from mediaai.audit_chain import AuditChain

__all__ = [
    "MediaVault",
    "MediaOptimizer",
    "FLEngine",
    "CellSegmentor",
    "ViabilityClassifier",
    "KnowledgeHub",
    "AuditChain",
]

__all__.extend([
    "__version__",
])