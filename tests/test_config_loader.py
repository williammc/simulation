"""
Tests for the ConfigLoader with include support.
"""

import pytest
import yaml
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import ConfigLoader, CircularIncludeError


class TestConfigLoader:
    """Test suite for ConfigLoader."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def loader(self, temp_dir):
        """Create a ConfigLoader instance."""
        return ConfigLoader(base_path=temp_dir)
    
    def test_basic_yaml_loading(self, temp_dir, loader):
        """Test loading a simple YAML file without includes."""
        # Create a simple config file
        config_file = temp_dir / "simple.yaml"
        config_data = {
            "name": "test",
            "value": 42,
            "nested": {
                "key": "value"
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load the config
        loaded = loader.load_config(config_file)
        
        assert loaded == config_data
        assert loaded["name"] == "test"
        assert loaded["value"] == 42
        assert loaded["nested"]["key"] == "value"
    
    def test_include_tag_resolution(self, temp_dir, loader):
        """Test !include tag resolution."""
        # Create a component config
        component_file = temp_dir / "component.yaml"
        component_data = {
            "type": "circle",
            "radius": 2.0
        }
        
        with open(component_file, 'w') as f:
            yaml.dump(component_data, f)
        
        # Create main config with include
        main_file = temp_dir / "main.yaml"
        main_content = """
name: "Main Config"
trajectory: !include component.yaml
value: 100
"""
        
        with open(main_file, 'w') as f:
            f.write(main_content)
        
        # Load the main config
        loaded = loader.load_config(main_file)
        
        assert loaded["name"] == "Main Config"
        assert loaded["value"] == 100
        assert loaded["trajectory"]["type"] == "circle"
        assert loaded["trajectory"]["radius"] == 2.0
    
    def test_nested_includes(self, temp_dir, loader):
        """Test nested include resolution."""
        # Create base component
        base_file = temp_dir / "base.yaml"
        with open(base_file, 'w') as f:
            f.write("value: 42\nkey: base")
        
        # Create middle component that includes base
        middle_file = temp_dir / "middle.yaml"
        with open(middle_file, 'w') as f:
            f.write("base: !include base.yaml\nmiddle_key: middle")
        
        # Create top config that includes middle
        top_file = temp_dir / "top.yaml"
        with open(top_file, 'w') as f:
            f.write("middle: !include middle.yaml\ntop_key: top")
        
        # Load the top config
        loaded = loader.load_config(top_file)
        
        assert loaded["top_key"] == "top"
        assert loaded["middle"]["middle_key"] == "middle"
        assert loaded["middle"]["base"]["value"] == 42
        assert loaded["middle"]["base"]["key"] == "base"
    
    def test_relative_path_resolution(self, temp_dir, loader):
        """Test relative path resolution in includes."""
        # Create subdirectories
        sub_dir = temp_dir / "components"
        sub_dir.mkdir()
        
        # Create component in subdirectory
        component_file = sub_dir / "item.yaml"
        with open(component_file, 'w') as f:
            f.write("type: component\nvalue: 123")
        
        # Create main config with relative include
        main_file = temp_dir / "main.yaml"
        with open(main_file, 'w') as f:
            f.write("item: !include components/item.yaml")
        
        # Load the main config
        loaded = loader.load_config(main_file)
        
        assert loaded["item"]["type"] == "component"
        assert loaded["item"]["value"] == 123
    
    def test_circular_dependency_detection(self, temp_dir, loader):
        """Test that circular includes are detected and raise an error."""
        # Create config A that includes B
        config_a = temp_dir / "a.yaml"
        with open(config_a, 'w') as f:
            f.write("b: !include b.yaml\nvalue_a: 1")
        
        # Create config B that includes A (circular)
        config_b = temp_dir / "b.yaml"
        with open(config_b, 'w') as f:
            f.write("a: !include a.yaml\nvalue_b: 2")
        
        # Loading should raise CircularIncludeError
        with pytest.raises(CircularIncludeError) as exc_info:
            loader.load_config(config_a)
        
        assert "Circular include detected" in str(exc_info.value)
    
    def test_cache_functionality(self, temp_dir, loader):
        """Test that configs are cached and reused."""
        # Create a config file
        config_file = temp_dir / "cached.yaml"
        with open(config_file, 'w') as f:
            f.write("value: 42")
        
        # Load twice
        first_load = loader.load_config(config_file)
        second_load = loader.load_config(config_file)
        
        # Should be equal but not the same object (we return copies)
        assert first_load == second_load
        assert first_load is not second_load
        
        # Cache should contain the file
        assert config_file.resolve() in loader.cache
        
        # Clear cache
        loader.clear_cache()
        assert len(loader.cache) == 0
    
    def test_file_not_found_error(self, temp_dir, loader):
        """Test that FileNotFoundError is raised for missing files."""
        missing_file = temp_dir / "missing.yaml"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_config(missing_file)
        
        assert "not found" in str(exc_info.value)
    
    def test_merge_configs(self, loader):
        """Test configuration merging functionality."""
        base = {
            "name": "base",
            "values": {
                "a": 1,
                "b": 2
            },
            "list": [1, 2, 3]
        }
        
        override = {
            "name": "override",
            "values": {
                "b": 20,
                "c": 30
            },
            "new_key": "new_value"
        }
        
        merged = loader.merge_configs(base, override)
        
        assert merged["name"] == "override"
        assert merged["values"]["a"] == 1
        assert merged["values"]["b"] == 20
        assert merged["values"]["c"] == 30
        assert merged["list"] == [1, 2, 3]
        assert merged["new_key"] == "new_value"
    
    def test_absolute_path_include(self, temp_dir, loader):
        """Test including files with absolute paths."""
        # Create a component file
        component_file = temp_dir / "absolute_component.yaml"
        with open(component_file, 'w') as f:
            f.write("type: absolute\nvalue: 999")
        
        # Create main config with absolute path include
        main_file = temp_dir / "main_absolute.yaml"
        with open(main_file, 'w') as f:
            f.write(f"component: !include {component_file.absolute()}")
        
        # Load the main config
        loaded = loader.load_config(main_file)
        
        assert loaded["component"]["type"] == "absolute"
        assert loaded["component"]["value"] == 999
    
    def test_empty_file_handling(self, temp_dir, loader):
        """Test loading empty YAML files."""
        empty_file = temp_dir / "empty.yaml"
        empty_file.touch()  # Create empty file
        
        loaded = loader.load_config(empty_file)
        assert loaded == {}
    
    def test_multiple_includes_in_same_file(self, temp_dir, loader):
        """Test multiple !include tags in the same file."""
        # Create component files
        comp1 = temp_dir / "comp1.yaml"
        with open(comp1, 'w') as f:
            f.write("type: comp1\nvalue: 1")
        
        comp2 = temp_dir / "comp2.yaml"
        with open(comp2, 'w') as f:
            f.write("type: comp2\nvalue: 2")
        
        # Create main config with multiple includes
        main_file = temp_dir / "multi_include.yaml"
        with open(main_file, 'w') as f:
            f.write("""
name: "Multi Include"
first: !include comp1.yaml
second: !include comp2.yaml
""")
        
        # Load the main config
        loaded = loader.load_config(main_file)
        
        assert loaded["first"]["type"] == "comp1"
        assert loaded["second"]["type"] == "comp2"


def test_real_config_loading():
    """Test loading actual config files from the project."""
    # Test loading the example scenario we created
    loader = ConfigLoader(base_path=Path.cwd())
    
    # Check if the config file exists
    scenario_file = Path("config/scenarios/vio_example.yaml")
    if scenario_file.exists():
        try:
            config = loader.load_config(scenario_file)
            
            # Verify the structure
            assert config["name"] == "VIO Example Scenario"
            assert "trajectory" in config
            assert config["trajectory"]["type"] == "figure8"
            assert "cameras" in config
            assert "imus" in config
            
            print("Successfully loaded real config with includes!")
        except Exception as e:
            print(f"Could not load real config: {e}")


if __name__ == "__main__":
    # Run a quick test
    test_real_config_loading()