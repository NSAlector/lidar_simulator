import os
import tomllib
import yaml

class ConfigLoader:
    @staticmethod
    def load(filepath: str) -> dict:
        """
        Loads configuration from a YAML or TOML file based on its extension.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
            
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        if ext in ('.yaml', '.yml'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif ext == '.toml':
            with open(filepath, 'rb') as f:
                return tomllib.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")
