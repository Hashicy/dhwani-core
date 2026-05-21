import sys
import subprocess
try:
    import librosa
    print("librosa imported successfully")
except Exception as e:
    print(f"Error importing librosa: {type(e).__name__}: {e}")
print("\nPython executable:", sys.executable)
print("\nsys.path:", sys.path)
print("\nInstalled packages:")
subprocess.run([sys.executable, "-m", "pip", "freeze"])
