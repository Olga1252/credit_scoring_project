$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

docker build -t credit-scoring-api $projectRoot

if ($LASTEXITCODE -ne 0) {
    throw "Docker build failed."
}




