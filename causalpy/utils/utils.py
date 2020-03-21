import os
import shutil


class TempFolder:
    def __init__(self, folder: str):
        self.folder = folder

    def __enter__(self):
        i = 0
        self.folder += f"_{i}"
        while os.path.isdir(self.folder):
            self.folder = self.folder.rsplit("_", 1)[0]
            i += 1
            self.folder += f"_{i}"

        os.mkdir(self.folder)
        return os.path.join(".", self.folder)

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.folder, ignore_errors=False)
