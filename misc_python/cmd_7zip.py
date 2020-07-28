

import subprocess
from pathlib import Path
import getpass

PASSWORD = getpass.getpass()

ZIP_EXE_PATH = Path("C:\\Program Files\\7-Zip\\7z.exe")

INPUT_ZIP_PATH = Path("C:\\Users\\T149900\\Downloads\\ziptest.7z")

OUTPUT_PATH = Path("C:\\Temp")

assert (ZIP_EXE_PATH.is_file())
assert (INPUT_ZIP_PATH.is_file())

OUTPUT_PATH.is_dir()

completed = subprocess.run([str(ZIP_EXE_PATH), "x", str(INPUT_ZIP_PATH), "*", f"-o{OUTPUT_PATH}", "-y", f"-p{PASSWORD}"], stdout=subprocess.PIPE)

zOutput = completed.stdout.decode('utf-8')

print (f"return code: {completed.returncode}")







