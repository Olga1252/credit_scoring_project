param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8001
)

$ErrorActionPreference = "Stop"

docker run -d --rm --name credit-scoring-api -p "127.0.0.1:${Port}:8000" credit-scoring-api

if ($LASTEXITCODE -ne 0) {
    throw "Container launch failed."
}

Write-Host "Swagger: http://127.0.0.1:$Port/docs"


