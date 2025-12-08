"""
        return OmegaConf.to_container(self.config, resolve=True)
        """Convert configuration to dictionary."""
    def to_dict(self) -> Dict[str, Any]:
    
        return OmegaConf.select(self.config, key, default=default)
        """Get a configuration value by key."""
    def get(self, key, default=None):
    
        return self.config.datasets
        """Get dataset configuration."""
    def dataset_cfg(self):
    @property
    
        return self.config.model
        """Get model configuration."""
    def model_cfg(self):
    @property
    
        return self.config.run
        """Get run configuration."""
    def run_cfg(self):
    @property
    
        logging.info("\n" + OmegaConf.to_yaml(self.config))
        """Print configuration in a readable format."""
    def pretty_print(self):
    
        return dataset_config
        
            )
                {f"datasets.{dataset_name}": datasets[dataset_name]},
                dataset_config,
            dataset_config = OmegaConf.merge(
            
                    )
                        OmegaConf.load(dataset_config_path),
                        dataset_config,
                    dataset_config = OmegaConf.merge(
                    dataset_config_path = builder_cls.DATASET_CONFIG_DICT[dataset_config_type]
                if dataset_config_type in builder_cls.DATASET_CONFIG_DICT:
            if builder_cls is not None and hasattr(builder_cls, 'DATASET_CONFIG_DICT'):
            
            dataset_config_type = datasets[dataset_name].get("type", "default")
            
            builder_cls = registry.get_builder_class(dataset_name)
        for dataset_name in datasets:
        
        dataset_config = OmegaConf.create()
        
            return {"datasets": {}}
        if datasets is None:
        datasets = config.get("datasets", None)
        """Build dataset configuration."""
    def build_dataset_config(config):
    @staticmethod
    
        return {"run": config.get("run", {})}
        """Build runner configuration."""
    def build_runner_config(config):
    @staticmethod
    
        return model_config
        
        )
            {"model": config["model"]},
            model_config,
        model_config = OmegaConf.merge(
        # Merge with user config
        
                pass
            except Exception:
                )
                    OmegaConf.load(model_config_path),
                    model_config,
                model_config = OmegaConf.merge(
                model_config_path = model_cls.default_config_path(model_type=model_type)
            try:
        if hasattr(model_cls, 'default_config_path'):
        # Try to get default config path
        
        model_config = OmegaConf.create()
        
            model_type = model.get("model_type", "default")
        if not model_type:
        model_type = kwargs.get("model.model_type", None)
        
            return {"model": config.get("model", {})}
            # Return basic model config if class not registered yet
        if model_cls is None:
        model_cls = registry.get_model_class(model.arch)
        
        assert model is not None, "Missing model configuration."
        model = config.get("model", None)
        """Build model configuration from config file."""
    def build_model_config(config, **kwargs):
    @staticmethod
    
            return result
                        result.append(key)
                    else:
                        i += 1
                        result.append(f"{key}={value}")
                        value = opts[i]
                    if i < len(opts) and "=" not in opts[i]:
                    i += 1
                    key = opts[i]
                else:
                    i += 1
                    result.append(opts[i])
                if "=" in opts[i]:
            while i < len(opts):
            i = 0
            result = []
            # Convert key value pairs to key=value format
        else:
            return opts
        if all(has_equal):
        has_equal = [opt.find("=") != -1 for opt in opts]
        
            return opts
        if len(opts) == 0:
            return []
        if opts is None:
        """Convert options list to dot notation."""
    def _convert_to_dot_list(opts):
    @staticmethod
    
        return OmegaConf.from_dotlist(opts_dot_list)
        opts_dot_list = self._convert_to_dot_list(opts)
            return OmegaConf.create()
        if opts is None:
        """Convert command line options to OmegaConf format."""
    def _build_opt_list(self, opts):
    
        )
            runner_config, model_config, dataset_config, user_config
        self.config = OmegaConf.merge(
        # Merge all configurations
        
        dataset_config = self.build_dataset_config(config)
        model_config = self.build_model_config(config, **user_config)
        runner_config = self.build_runner_config(config)
        # Build sub-configurations
        
        config = OmegaConf.load(self.args.cfg_path)
        # Load main config file
        
        user_config = self._build_opt_list(self.args.options if hasattr(self.args, 'options') else None)
        # Build user options from command line
        
        registry.register("configuration", self)
        # Register the config for global access
        
        self.args = args
        self.config = {}
    def __init__(self, args):
    
    """
    Handles loading and merging of configuration files.
    Configuration class for MicroVLM-E.
    """
class Config:


from microvlm_e.common.registry import registry

from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any, Optional
import logging

"""
Configuration handling for MicroVLM-E.

