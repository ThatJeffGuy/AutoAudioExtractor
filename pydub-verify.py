import sys
import logging

# Set up basic logging to console
logging.basicConfig(level=logging.INFO)

# Log Python executable and sys.path
logging.info(f"Python executable: {sys.executable}")
logging.info(f"sys.path: {sys.path}")

# Print Python executable and sys.path for direct feedback
print(f"Python executable: {sys.executable}")
print(f"sys.path: {sys.path}")

# Attempt to import pydub and log errors if any
try:
    from pydub import AudioSegment
    logging.info("pydub library loaded successfully")
    print("pydub library loaded successfully")
except ImportError as e:
    logging.error(f"Error importing pydub: {e}")
    logging.error(f"Current PYTHONPATH: {sys.path}")
    print(f"Error importing pydub: {e}")
    print(f"Current PYTHONPATH: {sys.path}")
    sys.exit(1)
