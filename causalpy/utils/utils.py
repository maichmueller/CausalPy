import os
import shutil


class TempFolder:
    def __init__(self, folder: str, **kwargs):
        self.folder = folder
        self.lock = kwargs.pop("LOCK", None)

    def lockit(self):
        if self.lock is not None:
            self.lock.acquire()

    def unlockit(self):
        if self.lock is not None:
            self.lock.release()

    def __enter__(self):
        self.lockit()
        i = 0
        self.folder += f"_{i}"
        while os.path.isdir(self.folder):
            self.folder = self.folder.rsplit("_", 1)[0]
            i += 1
            self.folder += f"_{i}"

        os.mkdir(self.folder)
        self.unlockit()
        return os.path.join(".", self.folder)

    def __exit__(self, type, value, traceback):
        self.lockit()
        shutil.rmtree(self.folder, ignore_errors=False)
        self.unlockit()
