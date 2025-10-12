# This script sends a sample request to the prediction endpoint using PowerShell.

Write-Host "Testing prediction endpoint with sample_payload.json..."

$headers = @{
    "Content-Type" = "application/json"
}
$body = Get-Content -Raw -Path "./tests/sample_payload.json"

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -Headers $headers -Body $body
