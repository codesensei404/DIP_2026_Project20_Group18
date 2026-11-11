"""
Master entry point for Pipeline 2.0 (Watershed Segmentation).
Delegates execution to Pipeline 2.0/main_II.py.
"""
import os
import sys
import importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_2_DIR = os.path.join(_THIS_DIR, "Pipeline 2.0")
MAIN_II_PATH = os.path.join(PIPELINE_2_DIR, "main_II.py")

spec = importlib.util.spec_from_file_location("pipeline2_main", MAIN_II_PATH)
pipeline2_main = importlib.util.module_from_spec(spec)
sys.modules["pipeline2_main"] = pipeline2_main
spec.loader.exec_module(pipeline2_main)

build_config = pipeline2_main.build_config
run_pipeline = pipeline2_main.run_pipeline
main = pipeline2_main.main

if __name__ == "__main__":
    main()
