import datetime

class Logger:
    def __init__(self, save_dir):
        self._logger_path = save_dir / "training.log"

        with open(self._logger_path, "w") as f:
            f.write(f"# Training started: {datetime.datetime.now()}\n")


    def _log(self, message: str):
        with open(self._logger_path, "a") as f:
            f.write(message + "\n")