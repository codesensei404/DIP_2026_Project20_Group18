"""
Master entry point for Pipeline 3.0 (Hybrid Threshold-Watershed Segmentation).
Delegates execution to Pipeline 3.0/main_III.py.
"""
import os
import sys
import importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_3_DIR = os.path.join(_THIS_DIR, "Pipeline 3.0")
MAIN_III_PATH = os.path.join(PIPELINE_3_DIR, "main_III.py")

spec = importlib.util.spec_from_file_location("pipeline3_main", MAIN_III_PATH)
pipeline3_main = importlib.util.module_from_spec(spec)
sys.modules["pipeline3_main"] = pipeline3_main
spec.loader.exec_module(pipeline3_main)

build_config = pipeline3_main.build_config
run_pipeline = pipeline3_main.run_pipeline
main = pipeline3_main.main

if __name__ == "__main__":
    main()
