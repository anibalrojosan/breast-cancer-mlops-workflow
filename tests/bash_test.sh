#!/usr/bin/env bash
set -euo pipefail

echo "Health check..."
curl -s http://127.0.0.1:5000/ | jq . || curl -s http://127.0.0.1:5000/

echo "Predict..."
curl -s -X POST -H "Content-Type: application/json" -d @tests/sample_payload.json http://127.0.0.1:5000/predict | jq . || curl -s -X POST -H "Content-Type: application/json" -d @tests/sample_payload.json http://127.0.0.1:5000/predict
