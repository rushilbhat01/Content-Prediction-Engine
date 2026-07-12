# yt_shorts -- dependency installer
# Run from the yt_shorts folder: .\install.ps1

Set-Location $PSScriptRoot

Write-Host "=== Step 0: ffmpeg (required by yt-dlp and librosa) ===" -ForegroundColor Cyan
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($ffmpeg) {
    Write-Host "ffmpeg already installed: $($ffmpeg.Source)" -ForegroundColor Green
} else {
    Write-Host "Installing ffmpeg via winget..." -ForegroundColor Yellow
    winget install Gyan.FFmpeg
    Write-Host "Restart this terminal after ffmpeg installs so it appears on PATH." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Step 1: pip install requirements ===" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "=== Step 2: torchvggish (GitHub) ===" -ForegroundColor Cyan
pip install git+https://github.com/harritaylor/torchvggish.git

Write-Host ""
Write-Host "=== Step 3: verify yt-dlp is on PATH ===" -ForegroundColor Cyan
yt-dlp --version
if (-not $?) {
    Write-Host "yt-dlp not on PATH -- try: pip install -U yt-dlp" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== All done! ===" -ForegroundColor Green
Write-Host "Next: fill in your API keys in .env"
