from cse.agent.tools import read_file

print("--- Reading src/cse/agent/tools.py itself ---")
print(read_file("src/cse/agent/tools.py"))

print("\n--- Reading a non-existent file ---")
print(read_file("does/not/exist.py"))

print("\n--- Attempting to escape the project directory ---")
print(read_file("../../etc/passwd"))
