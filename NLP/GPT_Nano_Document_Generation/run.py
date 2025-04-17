import sys
import os

# Add the current directory to the sys.path
# This setup ensures that the GPT directory is added to sys.path,
# allowing Python to find and import package1 and package2 correctly.
# The run.py script acts as a launcher that sets up the necessary environment
# for your package structure to work as expected.

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import and run the script
from training.train import main

if __name__ == "__main__":
    main()
