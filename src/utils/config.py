"""Configuration management"""
import yaml
from pathlib import Path
from typing import Dict, Any
import os

class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Expand environment variables
        config = self._expand_env_vars(config)
        return config
    
    def _expand_env_vars(self, obj: Any) -> Any:
        """Recursively expand environment variables in config"""
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        return obj
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self.get(key)
    
    @property
    def data(self) -> Dict[str, Any]:
        return self._config.get('data', {})
    
    @property
    def model(self) -> Dict[str, Any]:
        return self._config.get('model', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        return self._config.get('training', {})
    
    @property
    def mlflow(self) -> Dict[str, Any]:
        return self._config.get('mlflow', {})
    
    @property
    def api(self) -> Dict[str, Any]:
        return self._config.get('api', {})
    
    @property
    def ui(self) -> Dict[str, Any]:
        return self._config.get('ui', {})

