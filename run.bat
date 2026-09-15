@echo off
cd /d "%~dp0"
python main.py
if errorlevel 1 echo Pipeline failed. See the error above.
pause
