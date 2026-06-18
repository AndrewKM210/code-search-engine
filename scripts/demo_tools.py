from omegaconf import OmegaConf

from cse.agent.tools import grep, list_directory, read_file, search_code
from cse.search_engine.engine import CodeSearchEngine

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

print("\n--- Grepping for tool definitions in src/cse/agent ---")
print(grep(r"^def \w+", "src/cse/agent"))

print("\n--- Grepping for a pattern with no matches ---")
print(grep("this_will_not_match_anything", "src/cse/agent"))

print("\n--- Grepping with an invalid regex ---")
print(grep("(unclosed", "src/cse/agent"))

print("\n--- Attempting to escape the project directory ---")
print(grep("password", ".."))

config = OmegaConf.load("config/main_config.yaml")
engine = CodeSearchEngine(
    model_name=config.model_name,
    db_collection=config.qdrant.self_repo_collection,
    db_path=config.qdrant.storage_path,
    device=config.get("device", "auto"),
    quiet=True,
)

print("\n--- Searching the self-indexed repo for a coding concept ---")
print(search_code("how does resolve_device pick a compute device", engine))

print("\n--- Searching for something that won't match anything indexed ---")
print(search_code("kubernetes helm chart deployment", engine, k=2))
