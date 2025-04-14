# src/attack_models/factory.py
"""
Factory function to create instances of attack models.
"""

from typing import Dict, Type, Union, List
import logging

from .base import BaseAttackModel
# Import all implemented attack classes
from .implementations import _ALL_ATTACK_CLASSES

# Build the registry mapping IDs to classes
_ATTACK_MODEL_REGISTRY: Dict[str, Type[BaseAttackModel]] = {
    cls().attack_id: cls for cls in _ALL_ATTACK_CLASSES
}
# Add integer keys if needed for backward compatibility or convenience
_ATTACK_MODEL_REGISTRY.update({
    str(i): _ATTACK_MODEL_REGISTRY[str(i)] for i in range(13)
})
# Ensure 'ieee' key exists
if 'ieee' not in _ATTACK_MODEL_REGISTRY:
     logging.warning("AttackTypeIEEE implementation not found or attack_id mismatch.")


def get_attack_model(attack_id: Union[int, str]) -> BaseAttackModel:
    """
    Factory function to retrieve an instance of a specific attack model.

    Args:
        attack_id (Union[int, str]): The identifier of the attack model
                                     (e.g., 0, 12, 'ieee').

    Returns:
        BaseAttackModel: An instance of the requested attack model.

    Raises:
        ValueError: If the attack_id is not found in the registry.
    """
    attack_id_str = str(attack_id) # Normalize to string key

    model_class = _ATTACK_MODEL_REGISTRY.get(attack_id_str)

    if model_class is None:
        raise ValueError(f"Unknown attack_id: '{attack_id}'. Available IDs: {list(_ATTACK_MODEL_REGISTRY.keys())}")

    return model_class()

def list_available_attacks() -> List[str]:
    """Returns a list of available attack model IDs."""
    return sorted(_ATTACK_MODEL_REGISTRY.keys())
