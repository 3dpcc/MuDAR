import os
import shutil

from lightning.pytorch.callbacks import Callback


class ScriptBackupCallback(Callback):
    def __init__(self, script_paths):
        super().__init__()
        self.script_paths = script_paths

    def on_train_start(self, trainer, pl_module):
        os.mkdir(f"{trainer.log_dir}/backup")
        for script_path in self.script_paths:
            file_name = os.path.basename(script_path)
            backup_path = os.path.join(f"{trainer.log_dir}/backup", file_name)
            shutil.copy2(script_path, backup_path)
