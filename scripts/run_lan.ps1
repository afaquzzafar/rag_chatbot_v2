# ==============================================================================
# scripts/run_lan.ps1 -- start the chatbot so other devices on the same local
# network (Wi-Fi/office LAN) can open it at http://<this-PC-IP>:8501.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_lan.ps1
#
# The app has no login of its own: anyone on the network who opens the URL
# can use it (and your Gemini quota). Windows Firewall must allow inbound
# TCP 8501 for other devices to connect -- see README section 13.
# ==============================================================================

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = "M:\venvs\rag_chatbot\Scripts\python.exe"
# Reuse the model cache already downloaded to M: (keeps the C: drive free).
$env:HF_HOME = "M:\hf-cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$ip = (Get-NetIPAddress -AddressFamily IPv4 |
       Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254*' } |
       Select-Object -First 1).IPAddress
Write-Host "Other devices on this network can open: http://${ip}:8501"

& $python -m streamlit run frontend/app.py `
    --server.port=8501 --server.address=0.0.0.0 `
    --server.headless=true --browser.gatherUsageStats=false
