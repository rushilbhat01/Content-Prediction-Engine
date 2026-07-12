# Waits for fitness download to finish, then runs food automatically
Write-Host "Queue started. Waiting for fitness download to finish..."

# Wait until no python process is running
while (Get-Process python -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

Write-Host "Fitness done! Starting food download..."
Set-Location $PSScriptRoot
python redownload_extract.py --niche food

Write-Host "All done! Both fitness and food downloaded and extracted."
