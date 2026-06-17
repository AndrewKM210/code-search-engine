from cse.agent.tools import list_directory, read_file

print("--- Reading src/cse/agent/tools.py itself ---")
print(read_file("src/cse/agent/tools.py"))

print("\n--- Reading a non-existent file ---")
print(read_file("does/not/exist.py"))

print("\n--- Attempting to escape the project directory ---")
print(read_file("../../etc/passwd"))

print("\n--- Listing the project root ---")
print(list_directory("."))

print("\n--- Listing src/cse/agent ---")
print(list_directory("src/cse/agent"))

print("\n--- Listing a non-existent directory ---")
print(list_directory("does/not/exist"))

print("\n--- Attempting to escape the project directory ---")
print(list_directory("../.."))
