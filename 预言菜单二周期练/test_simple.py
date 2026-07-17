import sys
print(f"Python version: {sys.version}")
print("Hello World from test script")

import os
os.makedirs("test_data", exist_ok=True)
with open("test_data/test_output.txt", "w") as f:
    f.write("Test output\n")
print("File written successfully")