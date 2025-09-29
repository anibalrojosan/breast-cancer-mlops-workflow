$headers = @{ "Content-Type" = "application/json" }

Write-Host "Health check..."
Invoke-RestMethod -Uri "http://127.0.0.1:5000/" -Method Get

Write-Host "Predict..."
$body = Get-Content -Raw -Path "tests/sample_payload.json" | ConvertFrom-Json
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -Headers $headers -Body (ConvertTo-Json $body)
