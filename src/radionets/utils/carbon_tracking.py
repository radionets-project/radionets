from typing import Self

try:
    from codecarbon import OfflineEmissionsTracker

    _CODECARBON_AVAILABLE = True
except ImportError:
    _CODECARBON_AVAILABLE = False


__all__ = ["CarbonTracker"]


class DummyTracker:
    def start(self):
        pass

    def stop(self):
        pass


class CarbonTracker:
    def __init__(self, train_config, stop_inside_scope=True, *args, **kwargs):
        self.train_config = train_config
        self.use = _CODECARBON_AVAILABLE and train_config.logging.codecarbon
        self.stop = stop_inside_scope

    def __enter__(self) -> Self:
        if self.use:
            self.tracker = OfflineEmissionsTracker(
                **self.train_config.logging.codecarbon.model_dump()
            )
            self.tracker.start()
        else:
            self.tracker = DummyTracker()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.stop:
            self.tracker.stop()
        return None
