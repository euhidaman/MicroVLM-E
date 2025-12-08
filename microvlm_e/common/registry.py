"""
Registry for MicroVLM-E components.
Provides a central place to register and retrieve models, datasets, processors, etc.
"""


class Registry:
    """
    A central registry for all components in MicroVLM-E.
    """

    mapping = {
        "model_name_mapping": {},
        "task_name_mapping": {},
        "processor_name_mapping": {},
        "builder_name_mapping": {},
        "runner_name_mapping": {},
        "state": {},
        "paths": {},
    }

    @classmethod
    def register_model(cls, name):
        """Decorator to register a model class."""
        def wrap(model_cls):
            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(f"Model '{name}' already registered.")
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls
        return wrap

    @classmethod
    def register_task(cls, name):
        """Decorator to register a task class."""
        def wrap(task_cls):
            if name in cls.mapping["task_name_mapping"]:
                raise KeyError(f"Task '{name}' already registered.")
            cls.mapping["task_name_mapping"][name] = task_cls
            return task_cls
        return wrap

    @classmethod
    def register_processor(cls, name):
        """Decorator to register a processor class."""
        def wrap(processor_cls):
            if name in cls.mapping["processor_name_mapping"]:
                raise KeyError(f"Processor '{name}' already registered.")
            cls.mapping["processor_name_mapping"][name] = processor_cls
            return processor_cls
        return wrap

    @classmethod
    def register_builder(cls, name):
        """Decorator to register a dataset builder class."""
        def wrap(builder_cls):
            if name in cls.mapping["builder_name_mapping"]:
                raise KeyError(f"Builder '{name}' already registered.")
            cls.mapping["builder_name_mapping"][name] = builder_cls
            return builder_cls
        return wrap

    @classmethod
    def register_runner(cls, name):
        """Decorator to register a runner class."""
        def wrap(runner_cls):
            if name in cls.mapping["runner_name_mapping"]:
                raise KeyError(f"Runner '{name}' already registered.")
            cls.mapping["runner_name_mapping"][name] = runner_cls
            return runner_cls
        return wrap

    @classmethod
    def register(cls, name, obj):
        """Register an object with a given name."""
        cls.mapping["state"][name] = obj

    @classmethod
    def register_path(cls, name, path):
        """Register a path with a given name."""
        cls.mapping["paths"][name] = path

    @classmethod
    def get_model_class(cls, name):
        """Get a registered model class by name."""
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_task_class(cls, name):
        """Get a registered task class by name."""
        return cls.mapping["task_name_mapping"].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        """Get a registered processor class by name."""
        return cls.mapping["processor_name_mapping"].get(name, None)

    @classmethod
    def get_builder_class(cls, name):
        """Get a registered builder class by name."""
        return cls.mapping["builder_name_mapping"].get(name, None)

    @classmethod
    def get_runner_class(cls, name):
        """Get a registered runner class by name."""
        return cls.mapping["runner_name_mapping"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        """Get a registered object by name."""
        return cls.mapping["state"].get(name, default)

    @classmethod
    def get_path(cls, name):
        """Get a registered path by name."""
        return cls.mapping["paths"].get(name, None)

    @classmethod
    def list_models(cls):
        """List all registered model names."""
        return list(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def list_tasks(cls):
        """List all registered task names."""
        return list(cls.mapping["task_name_mapping"].keys())

    @classmethod
    def list_processors(cls):
        """List all registered processor names."""
        return list(cls.mapping["processor_name_mapping"].keys())

    @classmethod
    def list_builders(cls):
        """List all registered builder names."""
        return list(cls.mapping["builder_name_mapping"].keys())

    @classmethod
    def list_runners(cls):
        """List all registered runner names."""
        return list(cls.mapping["runner_name_mapping"].keys())


registry = Registry()

