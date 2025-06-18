"""
Configuration file for the hypothesis testing project.

This module contains all configuration settings, file paths,
and constants used throughout the project.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base data directory
BASE_DATA_DIR = Path("../data_folder")

# Data file paths configuration
DATA_PATHS = {
    # Main data files (20250127 folder)
    "lp_history": BASE_DATA_DIR / "20250127" / "20250127" / "LPヒストリー_hashed.csv",
    "salary": BASE_DATA_DIR / "20250127" / "20250127" / "報酬データ_hashed.csv",
    "performance": BASE_DATA_DIR / "20250127" / "20250127" / "業績_hashed.csv",
    "punishment": BASE_DATA_DIR / "20250127" / "20250127" / "懲戒処分_事故区分等追加_hashed.csv",
    "president_cup": BASE_DATA_DIR / "20250127" / "20250127" / "社長杯入賞履歴_LPコード0埋_hashed.csv",
    "jimu_miss": BASE_DATA_DIR / "20250127" / "20250127" / "★事務ミスデータ_不要データ削除版_hashed.xlsx",
    "complaint": BASE_DATA_DIR / "20250127" / "20250127" / "苦情データ_hashed.xlsx",
    "u24_247_1": BASE_DATA_DIR / "20250127" / "20250127" / "修正後_【U24-247】抽出結果①20191001~20211231_hashed.xlsx",
    "u24_247_2": BASE_DATA_DIR / "20250127" / "20250127" / "修正後_【U24-247】抽出結果②20220101~20240930_hashed.xlsx",
    "meeting_attendance": BASE_DATA_DIR / "20250127" / "20250127" / "MTG出席率2021-2023_hashed.csv",

    # Answer record files (data_0806 folder)
    "answer_record": BASE_DATA_DIR / "data_0806" / "回答記録_2024分析_hashed.xlsx",
}

# Column mapping configuration
COLUMN_MAPPINGS = {
    "lp_columns": ["LP", "MGR", "AMGR"],
    "office_columns": ["OFFICE"],
    "date_columns": {
        "lp_history": {"year": "JOB_YYY", "month": "JOB_MM", "day": "JOB_DD"},
        "salary": {"year": "S_YR", "month": "S_MO"},
        "performance": {"year_month": "ym"},
    }
}

# Data processing constants
PROCESSING_CONSTANTS = {
    "year_offset": 1800,  # Year offset for date conversion
    "filter_year_start": 2011,  # Start year for data filtering
    "u24_247_filter_year": 2021,  # Start year for U24-247 data
}

# Output configuration
OUTPUT_CONFIG = {
    "enable_warnings": False,  # Whether to show warnings
    "date_format": "%Y-%m-%d",
    "datetime_format": "%Y-%m-%d %H:%M:%S",
}


def get_data_path(key: str) -> Path:
    """
    Get the full path for a data file.

    Args:
        key: The key for the data file in DATA_PATHS

    Returns:
        Path object for the data file

    Raises:
        KeyError: If the key is not found in DATA_PATHS
        FileNotFoundError: If the file does not exist
    """
    if key not in DATA_PATHS:
        raise KeyError(f"Data path key '{key}' not found in configuration")

    path = DATA_PATHS[key]
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    return path


def validate_data_paths() -> Dict[str, bool]:
    """
    Validate that all configured data paths exist.

    Returns:
        Dictionary mapping file keys to existence status
    """
    validation_results = {}
    for key, path in DATA_PATHS.items():
        validation_results[key] = path.exists()

    return validation_results


def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of the current configuration.

    Returns:
        Dictionary containing configuration summary
    """
    return {
        "base_data_dir": str(BASE_DATA_DIR),
        "total_data_files": len(DATA_PATHS),
        "processing_constants": PROCESSING_CONSTANTS,
        "output_config": OUTPUT_CONFIG,
    }
