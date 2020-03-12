

from mp4_frames import get_meta_dir
from datetime import datetime
from pathlib import Path

class Logger:
    def write(self, zTxt):
        with self._log_path.open("a") as f:
            f.write(zTxt + "\n")

    def __init__(self, zFile):  
        output_dir = get_meta_dir()
        assert output_dir.is_dir()
        
        zTime = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self._log_path = output_dir / (zFile + "_" + zTime + ".txt")


