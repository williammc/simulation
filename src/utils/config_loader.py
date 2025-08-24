"""
Enhanced configuration loader with file inclusion support.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Set, Optional
from functools import lru_cache


class CircularIncludeError(Exception):
    """Raised when circular dependencies are detected in config includes."""
    pass


class ConfigLoader:
    """Configuration loader with support for !include tags and caching."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the config loader.
        
        Args:
            base_path: Base directory for resolving relative paths.
                      Defaults to current working directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.cache: Dict[Path, Dict] = {}
        self._include_stack: Set[Path] = set()
        self._setup_yaml_loader()
    
    def _setup_yaml_loader(self):
        """Setup custom YAML loader with !include support."""
        # Create a custom loader class
        class IncludeLoader(yaml.SafeLoader):
            pass
        
        # Store reference to self for use in constructor
        loader_self = self
        
        def include_constructor(loader, node):
            """Handle !include tag."""
            # Get the file path from the node
            file_path = loader.construct_scalar(node)
            
            # Get the current file's directory
            if hasattr(loader, 'current_file'):
                current_dir = Path(loader.current_file).parent
            else:
                current_dir = loader_self.base_path
            
            # Resolve the include path
            resolved_path = loader_self._resolve_path(file_path, current_dir)
            
            # Load the included file
            return loader_self._load_file(resolved_path)
        
        # Add the constructor to our custom loader
        IncludeLoader.add_constructor('!include', include_constructor)
        
        # Store the loader class
        self.yaml_loader = IncludeLoader
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a configuration file with includes resolved.
        
        Args:
            config_path: Path to the configuration file.
        
        Returns:
            Loaded configuration dictionary.
        
        Raises:
            FileNotFoundError: If config file doesn't exist.
            CircularIncludeError: If circular includes are detected.
        """
        config_path = Path(config_path)
        
        # Resolve to absolute path
        if not config_path.is_absolute():
            config_path = self.base_path / config_path
        
        config_path = config_path.resolve()
        
        # Check if file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Clear the include stack for this load operation
        self._include_stack.clear()
        
        # Load the configuration
        return self._load_file(config_path)
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a YAML file with caching and circular dependency detection.
        
        Args:
            file_path: Path to the YAML file.
        
        Returns:
            Loaded configuration dictionary.
        
        Raises:
            CircularIncludeError: If circular includes are detected.
        """
        file_path = file_path.resolve()
        
        # Check for circular dependencies
        if file_path in self._include_stack:
            cycle = ' -> '.join(str(p) for p in self._include_stack)
            cycle += f' -> {file_path}'
            raise CircularIncludeError(f"Circular include detected: {cycle}")
        
        # Check cache
        if file_path in self.cache:
            return self.cache[file_path].copy()
        
        # Add to include stack
        self._include_stack.add(file_path)
        
        try:
            # Load the file
            with open(file_path, 'r') as f:
                # Read the content first
                content = f.read()
            
            # Create a loader instance and set the current file
            loader = self.yaml_loader(content)
            loader.current_file = file_path
            
            try:
                # Load the YAML content
                config = loader.get_single_data()
            finally:
                loader.dispose()
            
            if config is None:
                config = {}
            
            # Cache the result
            self.cache[file_path] = config
            
            return config.copy()
        
        finally:
            # Remove from include stack
            self._include_stack.discard(file_path)
    
    def _resolve_path(self, path: Union[str, Path], relative_to: Path) -> Path:
        """
        Resolve a path relative to a given directory.
        
        Args:
            path: Path to resolve (can be relative or absolute).
            relative_to: Directory to resolve relative paths against.
        
        Returns:
            Resolved absolute path.
        """
        path = Path(path)
        
        # If absolute, return as is
        if path.is_absolute():
            return path.resolve()
        
        # Try relative to the given directory first
        resolved = (relative_to / path).resolve()
        if resolved.exists():
            return resolved
        
        # Try relative to base path
        resolved = (self.base_path / path).resolve()
        if resolved.exists():
            return resolved
        
        # Return the first resolution attempt even if it doesn't exist
        # (will fail with better error message when trying to load)
        return (relative_to / path).resolve()
    
    def clear_cache(self):
        """Clear the configuration cache."""
        self.cache.clear()
    
    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        
        Args:
            base: Base configuration.
            override: Configuration to override with.
        
        Returns:
            Merged configuration.
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                result[key] = self.merge_configs(result[key], value)
            else:
                # Override the value
                result[key] = value
        
        return result


# Convenience functions for backward compatibility
_default_loader = None

def get_default_loader() -> ConfigLoader:
    """Get or create the default config loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = ConfigLoader()
    return _default_loader


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a configuration file using the default loader.
    
    Args:
        config_path: Path to the configuration file.
    
    Returns:
        Loaded configuration dictionary.
    """
    return get_default_loader().load_config(config_path)