"""
Hypothesis Testing Module

This module contains all hypothesis testing and statistical analysis
functionality for validating research hypotheses. Each hypothesis
has its own dedicated folder for specific implementations.
"""

# Import hypothesis-specific validators
try:
    from .hypothesis_1.hypothesis_1 import Hypothesis1ValidatorCorrect as Hypothesis1Validator
except ImportError:
    Hypothesis1Validator = None

try:
    from .hypothesis_2.hypothesis_2 import Hypothesis2Validator
except ImportError:
    Hypothesis2Validator = None

# Add other hypothesis imports as needed

__all__ = [
    'Hypothesis1Validator',
    'Hypothesis2Validator',
]
