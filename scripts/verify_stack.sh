#!/usr/bin/env bash
# Verifikasi sistem docker-compose: status kontainer, network, volume,
# dan komunikasi antar kontainer. Jalankan dari root repo setelah `docker compose up -d`.

set -u
API_URL="http://localhost:8000"

section() {
  echo
  echo "=================================================================="
  echo " $1"
  echo "=================================================================="
}

section "1. Status kontainer (docker compose ps -a)"
docker compose ps -a

section "2. Kontainer di custom bridge network 'mlops-network'"
docker network inspect mlops-network \
  --format '{{range .Containers}}{{.Name}}  ->  {{.IPv4Address}}{{"\n"}}{{end}}'

section "3. Named volume untuk data persisten"
docker volume ls --filter name=mlops-

section "4. Service discovery: api-service memanggil mlflow-server lewat nama service"
docker compose exec -T api-service getent hosts mlflow-server
echo -n "GET http://mlflow-server:5000/health dari dalam api-service -> "
docker compose exec -T api-service curl -s http://mlflow-server:5000/health
echo

section "5. Model yang dimuat API (metadata diambil dari MLflow server)"
curl -s "$API_URL/model-info" | python3 -m json.tool

section "6. Prediksi lewat API"
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d @scripts/sample_request.json | python3 -m json.tool

section "7. Log api-service: model ditarik dari tracking server"
docker compose logs api-service | grep -i "berhasil ditarik"

section "8. Log mlflow-server: request unduh artefak model dari api-service"
docker compose logs mlflow-server | grep "mlflow-artifacts" | grep "GET" | tail -n 5
