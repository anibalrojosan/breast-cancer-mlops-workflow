#!/bin/bash
# This script sends a sample request to the prediction endpoint.

echo "Testing prediction endpoint with sample_payload.json..."

curl -X POST \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_payload.json \
  http://127.0.0.1:5000/predict

echo -e "\nDone."
