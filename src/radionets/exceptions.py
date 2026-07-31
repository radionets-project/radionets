class OptionalDependencyMissing(ModuleNotFoundError):
    def __init__(self, dependency):
        exc_msg = "To use this functionality, you need to install radionets "
        exc_msg += f"with the [{dependency}] optional dependency"

        super().__init__(exc_msg)
