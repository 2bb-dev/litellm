"""
Cost calculator for DeepSeek Chat models.

Handles prompt caching scenario.
"""

from datetime import datetime
from typing import Optional, Tuple, Union

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage


def cost_per_token(
    model: str, usage: Usage, request_time: Optional[Union[datetime, float]] = None
) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Follows the same logic as Anthropic's cost per token calculation.
    """
    return generic_cost_per_token(model=model, usage=usage, custom_llm_provider="deepseek", request_time=request_time)
