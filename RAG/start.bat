@echo off
echo 正在启动企业知识库RAG系统...
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
pause
