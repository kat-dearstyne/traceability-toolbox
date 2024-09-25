from toolbox.traceability.relationship_manager.cross_encoder_manager import CrossEncoderManager
from toolbox.traceability.relationship_manager.embeddings_manager import EmbeddingsManager
from toolbox.util.supported_enum import SupportedEnum


class SupportedRelationshipManager(SupportedEnum):
    EMBEDDING = EmbeddingsManager
    CROSS_ENCODER = CrossEncoderManager
