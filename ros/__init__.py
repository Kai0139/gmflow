import sys
from pathlib import Path
gmflow_path = Path(__file__).resolve().parent.parent
print(gmflow_path)
sys.path.append(str(gmflow_path))
