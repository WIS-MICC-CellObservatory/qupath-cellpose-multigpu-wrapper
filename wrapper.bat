@echo off
REM Only needed if you're running the .py directly instead of a compiled
REM .exe (some QuPath preference fields require an actual executable and
REM won't accept a bare .py file). Edit the path below, then point
REM QuPath's Cellpose python-executable preference at THIS .bat instead
REM of at the .py file.
python "C:\path\to\multi_gpu_cellpose_wrapper.py" %*
