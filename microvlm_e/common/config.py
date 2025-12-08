"""
Configuration handling for MicroVLM-E.
"""

import logging
from typing import Dict, Any

from omegaconf import OmegaConf, DictConfig

from microvlm_e.common.registry import registry


class Config:
    """
    Configuration class for MicroVLM-E.
    Handles loading and merging of configuration files.
    """

    def __init__(self, args):
        self.args = args
        self.config = {}

        # Register the config for global access
        registry.register("configuration", self)

        # Build user options from command line
        user_config = self._build_opt_list(self.args.options if hasattr(self.args, 'options') else None)

        # Load main config file
        config = OmegaConf.load(self.args.cfg_path)

        # Build sub-configurations
        runner_config = self.build_runner_config(config)
        model_config = self.build_model_config(config, **user_config)
        dataset_config = self.build_dataset_config(config)

        # Merge all configurations
        self.config = OmegaConf.merge(
            runner_config, model_config, dataset_config, user_config
        )

    def _build_opt_list(self, opts):
        """Convert command line options to OmegaConf format."""
        if opts is None:
            return OmegaConf.create()
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    @staticmethod
    def _convert_to_dot_list(opts):
        """Convert options list to dot notation."""
        if opts is None:
            return []
        if len(opts) == 0:
            return opts

        has_equal = [opt.find("=") != -1 for opt in opts]
        if all(has_equal):
            return opts
        else:
            result = []
            i = 0
            while i < len(opts):
                if "=" in opts[i]:
                    result.append(opts[i])
                    i += 1
                else:
                    key = opts[i]
                    i += 1
                    if i < len(opts) and "=" not in opts[i]:
                        value = opts[i]
                        result.append(f"{key}={value}")
                        i += 1
                    else:
                        result.append(key)
            return result

    @staticmethod
    def build_model_config(config, **kwargs):
        """Build model configuration from config file."""
        model = config.get("model", None)
        assert model is not None, "Missing model configuration."

        model_cls = registry.get_model_class(model.arch)
        if model_cls is None:
            return {"model": config.get("model", {})}

        model_type = kwargs.get("model.model_type", None)
        if not model_type:
            model_type = model.get("model_type", "default")

        model_config = OmegaConf.create()

        if hasattr(model_cls, 'default_config_path'):
            try:
                model_config_path = model_cls.default_config_path(model_type=model_type)
                model_config = OmegaConf.merge(
                    model_config,
                    OmegaConf.load(model_config_path),
                )
            except Exception:
                pass

        model_config = OmegaConf.merge(
            model_config,
            {"model": config["model"]},
        )

        return model_config

    @staticmethod
    def build_runner_config(config):
        """Build runner configuration."""
        return {"run": config.get("run", {})}

    @staticmethod
    def build_dataset_config(config):
        """Build dataset configuration."""
        datasets = config.get("datasets", None)
        if datasets is None:
            return {"datasets": {}}

        dataset_config = OmegaConf.create()

        for dataset_name in datasets:
            builder_cls = registry.get_builder_class(dataset_name)

            dataset_config_type = datasets[dataset_name].get("type", "default")

            if builder_cls is not None and hasattr(builder_cls, 'DATASET_CONFIG_DICT'):
                if dataset_config_type in builder_cls.DATASET_CONFIG_DICT:
                    dataset_config_path = builder_cls.DATASET_CONFIG_DICT[dataset_config_type]
                    dataset_config = OmegaConf.merge(
                        dataset_config,
                        OmegaConf.load(dataset_config_path),
                    )

            dataset_config = OmegaConf.merge(
                dataset_config,
                {f"datasets.{dataset_name}": datasets[dataset_name]},
            )

        return dataset_config

    def pretty_print(self):
        """Print configuration in a readable format."""
        logging.info("\n" + OmegaConf.to_yaml(self.config))

    @property
    def run_cfg(self):
        """Get run configuration."""
        return self.config.run

    @property
    def model_cfg(self):
        """Get model configuration."""
        return self.config.model

    @property
    def dataset_cfg(self):
        """Get dataset configuration."""
        return self.config.datasets

    def get(self, key, default=None):
        """Get a configuration value by key."""
        return OmegaConf.select(self.config, key, default=default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self.config, resolve=True)

