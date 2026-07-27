@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set BACKEND_DIR=C:\Users\Windows\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a62073cba53ad7a6054aa00\staging\enterprise-rag-kb\backend
cd /d "%BACKEND_DIR%"
"C:\Users\Windows\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\vm\tools\python\python.exe" -m scripts.run_ablation
