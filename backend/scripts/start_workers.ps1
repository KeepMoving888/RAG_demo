# Enterprise RAG Knowledge Base - 多 Worker 生产启动脚本
# 用法: powershell -ExecutionPolicy Bypass -File scripts/start_workers.ps1
# 默认 4 Worker (可通过环境变量 UVICORN_WORKERS 调整)
Set-Location "C:\Users\Windows\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a62073cba53ad7a6054aa00\staging\enterprise-rag-kb\backend"

# 从 .env 读取 Worker 数量, 默认 4
$envFile = ".env"
$workers = 4
if (Test-Path $envFile) {
    $line = Get-Content $envFile | Where-Object { $_ -match "^UVICORN_WORKERS=" } | Select-Object -First 1
    if ($line) {
        $workers = [int]($line.Split("=")[1].Trim())
    }
}
# 环境变量优先
if ($env:UVICORN_WORKERS) { $workers = [int]$env:UVICORN_WORKERS }

Write-Host "=== Enterprise RAG Knowledge Base - 多 Worker 启动 ===" -ForegroundColor Cyan
Write-Host "Worker 数量: $workers" -ForegroundColor Green
Write-Host "端口: 8765" -ForegroundColor Green
Write-Host "模式: 生产 (无 --reload)" -ForegroundColor Green
Write-Host ""

# uvicorn 多 Worker (Windows 上用 spawn, 绕开 GIL 线性提升 QPS)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8765 --workers $workers
