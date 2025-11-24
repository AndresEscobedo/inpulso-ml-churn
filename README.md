# ML Churn Deployment Project

This repository packages everything needed to train, evaluate, and deploy churn (employee attrition) models with FastAPI, Docker, Docker Compose, Minikube, and Google Kubernetes Engine (GKE). Traffic is split between a main model and a canary model by a dedicated Elector service that honors configurable weights.

## Architecture diagram

```mermaid
graph LR
    subgraph Data & Training
        T[HR Attrition Dataset]
        R[training/train_churn_models.py]
        T --> R
        R --> M[(artifacts/main_model.joblib)]
        R --> C[(artifacts/canary_model.joblib)]
        R --> REG[(artifacts/model_registry.json)]
    end

    subgraph Serving
        subgraph Main Model
            A[FastAPI Main API]
            B[Random Forest Pipeline]
            A --> B
        end

        subgraph Canary Model
            D[FastAPI Canary API]
            E[Gradient Boosting Pipeline]
            D --> E
        end

        subgraph Elector Service
            F[FastAPI Elector]
            G[Weighted Router]
            F --> G
            G -->|default 80%| A
            G -->|default 20%| D
        end
    end

    Client[/Client Application/] --> F
    A --> F
    D --> F
```

## Project Overview

- **Main model** (`main_model/`): Random Forest churn pipeline exported as `artifacts/main_model.joblib`.
- **Canary model** (`canary_model/`): Gradient Boosting challenger exported as `artifacts/canary_model.joblib`.
- **Elector service** (`elector/`): FastAPI router that sends each request to either model based on `CANARY_WEIGHT` and retries on failure.
- **Shared utilities** (`common/`): Feature schema, preprocessing helpers, and centralized path definitions.
- **Kubernetes manifests** (`k8s/`): Deployments, services, and namespace resources for Minikube or GKE.
- **Artifacts & registry** (`artifacts/`): Persisted models plus `model_registry.json` with metrics to guide rollout decisions.

## Repository Structure

```
├── artifacts/
│   ├── canary_model.joblib           # Challenger pipeline
│   ├── main_model.joblib             # Primary pipeline
│   └── model_registry.json           # Metrics logged after training
├── canary_model/
│   ├── app.py                        # FastAPI service exposing /predict
│   └── Dockerfile                    # Container for canary API
├── common/
│   ├── churn_config.py               # Paths, feature lists, constants
│   ├── model_utils.py                # Artifact loading + preprocessing
│   └── schemas.py                    # Pydantic request schema
├── elector/
│   ├── app.py                        # Weighted router with retries
│   └── Dockerfile
├── k8s/
│   ├── namespace.yaml
│   ├── model-deployment.yaml
│   ├── model-svc.yaml
│   ├── canary-deployment.yaml
│   ├── canary-svc.yaml
│   ├── elector-deployment.yaml
│   └── elector-svc.yaml
├── main_model/
│   ├── app.py
│   └── Dockerfile
├── training/
│   └── train_churn_models.py         # Retraining entrypoint
├── docker-compose.yml                # Local multi-service run
├── requirements.txt
└── README.md
```

## Model Lifecycle

1. **Dataset**: `HR_Employee_Attrition_Dataset con faltantes.csv` contains categorical and numeric HR attributes with the `attrition` target.
2. **Preprocessing**: Numeric features are imputed + scaled, categorical features are imputed + one-hot encoded (see `common/churn_config.py`).
3. **Training**: `training/train_churn_models.py` fits two pipelines (Random Forest for main, Gradient Boosting for canary), saves artifacts, and updates `artifacts/model_registry.json` with metrics.
4. **Serving**: FastAPI apps load the serialized pipelines, expose `/predict`, `/health`, and `/metadata`, and return churn probabilities.

### Training or refreshing the models

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m training.train_churn_models
```

Artifacts are written to `artifacts/` by default. Override `MODEL_ARTIFACT_PATH` if you store models elsewhere.

### Feature schema

Key request fields include: `age`, `businesstravel`, `dailyrate`, `department`, `distancefromhome`, `education`, `educationfield`, `environmentsatisfaction`, `gender`, `hourlyrate`, `jobinvolvement`, `joblevel`, `jobrole`, `disobediencerules`, `jobsatisfaction`, `maritalstatus`, `monthlyincome`, `monthlyrate`, `numcompaniesworked`, `overtime`, `percentsalaryhike`, `performancerating`, `relationshipsatisfaction`, `stockoptionlevel`, `totalworkingyears`, `trainingtimeslastyear`, `worklifebalance`, `yearsatcompany`, `yearsincurrentrole`, `yearssincelastpromotion`, `yearswithcurrmanager`.

## Getting Started

### Prerequisites

- Google Cloud Platform project with billing enabled.
- Google Cloud SDK (`gcloud`).
- Docker Engine + Docker Compose plugin.
- Python 3.9+ with `pip` and `venv`.
- kubectl and Minikube (for local clusters).
- Artifact Registry API enabled if you push images to GCP.

### Setting Up GCP VM (optional but recommended)

```bash
export PROJECT_ID=inbest-transformation
export LOCATION=us-central1
export REPOSITORY=iNPulso-ds

gcloud compute instances create ml-deployment-vm \
  --project=$PROJECT_ID \
  --zone=$LOCATION-a \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-type=pd-ssd \
  --boot-disk-size=50GB \
  --scopes=https://www.googleapis.com/auth/cloud-platform

gcloud compute instances add-labels ml-deployment-vm \
  --project=$PROJECT_ID \
  --zone=$LOCATION-a \
  --labels=environment=development,project=inpulso-ds-churn

gcloud compute firewall-rules create allow-minikube-dashboard \
  --project=$PROJECT_ID \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:30000-32767 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=ml-deployment-vm

gcloud compute instances add-tags ml-deployment-vm \
  --project=$PROJECT_ID \
  --zone=$LOCATION-a \
  --tags=ml-deployment-vm

gcloud compute ssh ml-deployment-vm --project=$PROJECT_ID --zone=$LOCATION-a
```

### Installing Dependencies

```bash
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker

sudo apt-get install -y python3 python3-pip python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Running Locally (uvicorn processes)

1. Clone the repo and enter it:
   ```bash
   git clone https://github.com/AndresEscobedo/kubemini-test-ml.git
   cd kubemini-test-ml
   ```

2. Activate your virtual environment and generate artifacts if needed:
   ```bash
   source .venv/bin/activate  # or create it if missing
   python -m training.train_churn_models  # optional when artifacts already exist
   ```

3. Start each FastAPI service (use separate terminals or a process manager):
   ```bash
   uvicorn main_model.app:app --host 0.0.0.0 --port 5000 --reload
   uvicorn canary_model.app:app --host 0.0.0.0 --port 5001 --reload
   CANARY_WEIGHT=35 uvicorn elector.app:app --host 0.0.0.0 --port 5002 --reload
   ```

4. Test the routed endpoint:
   ```bash
   curl -X POST "http://localhost:5002/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 34,
       "monthlyincome": 4500,
       "department": "Research & Development",
       "jobrole": "Laboratory Technician",
       "overtime": "No",
       "businesstravel": "Travel_Rarely"
     }'
   ```

### Running with Docker Compose

1. Confirm `artifacts/main_model.joblib` and `artifacts/canary_model.joblib` exist (run the training script if not).

2. Build and start all services:
   ```bash
   docker compose up --build
   ```

3. Override the routing weight when needed:
   ```bash
   CANARY_WEIGHT=10 docker compose up elector
   ```

4. Call `http://localhost:5002/predict` with the churn payload used earlier.

## Kubernetes Deployment

### Setting Up Minikube

```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
minikube start --driver=docker --cpus=4 --memory=8g --disk-size=20g
kubectl get nodes
```

### Building and loading images for Minikube

```bash
docker build -t churn-main:latest -f main_model/Dockerfile .
docker build -t churn-canary:latest -f canary_model/Dockerfile .
docker build -t churn-elector:latest -f elector/Dockerfile .
minikube image load churn-main:latest
minikube image load churn-canary:latest
minikube image load churn-elector:latest
```

Update the deployment manifests to reference these local tags and set `imagePullPolicy: Never` when using Minikube's Docker daemon.

### Deploying to Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/model-deployment.yaml -n churn
kubectl apply -f k8s/model-svc.yaml -n churn
kubectl apply -f k8s/canary-deployment.yaml -n churn
kubectl apply -f k8s/canary-svc.yaml -n churn
kubectl apply -f k8s/elector-deployment.yaml -n churn
kubectl apply -f k8s/elector-svc.yaml -n churn
kubectl get deployments -n churn
kubectl get pods -n churn
kubectl get svc -n churn
```

### Testing the deployment

```bash
NODE_PORT=$(kubectl get svc elector -n churn -o jsonpath='{.spec.ports[0].nodePort}')
ELECTOR_IP=$(minikube ip)
curl -X POST "http://$ELECTOR_IP:$NODE_PORT/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 40, "monthlyincome": 5200, "overtime": "Yes"}'
```

Use `kubectl logs deployment/elector -n churn` if routing or retries misbehave, and `minikube service elector -n churn --url` to obtain a tunnelled URL when NodePort access is blocked.

### Adjusting routing weights in-cluster

Edit `k8s/elector-deployment.yaml` and change the `CANARY_WEIGHT` environment variable. Re-apply and wait for the rollout:

```bash
kubectl apply -f k8s/elector-deployment.yaml -n churn
kubectl rollout status deployment/elector -n churn
curl http://$ELECTOR_IP:$NODE_PORT/health  # confirms active weight
```

## Deploying to Google Cloud Platform (GCP)

### 1. Environment variables

```bash
export PROJECT_ID=inbest-transformation
export LOCATION=us-central1
export REPOSITORY=inpulso-ds
export CLUSTER_NAME=inpulso-ds
```

### 2. Create the GKE cluster

```bash
gcloud services enable container.googleapis.com
gcloud container clusters create $CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$LOCATION-a \
  --machine-type=e2-standard-2 \
  --num-nodes=3 \
  --disk-size=80 \
  --disk-type=pd-standard \
  --release-channel=regular
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$LOCATION-a --project=$PROJECT_ID
kubectl get nodes
```

### 3. Configure Artifact Registry

```bash
gcloud services enable artifactregistry.googleapis.com
gcloud artifacts repositories create $REPOSITORY \
  --repository-format=docker \
  --location=$LOCATION \
  --description="Churn model containers"
gcloud auth configure-docker $LOCATION-docker.pkg.dev
```

### 4. Build and push container images (Cloud Build)

```bash
IMAGE_TAG=latest
MAIN_IMAGE=$LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/main-model:$IMAGE_TAG
CANARY_IMAGE=$LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/canary-model:$IMAGE_TAG
ELECTOR_IMAGE=$LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/elector:$IMAGE_TAG

gcloud services enable cloudbuild.googleapis.com

gcloud builds submit \
  --config main_model/cloudbuild.yaml \
  --substitutions=_REGION=$LOCATION,_REPOSITORY=$REPOSITORY,_IMAGE_TAG=$IMAGE_TAG \
  .

gcloud builds submit \
  --config canary_model/cloudbuild.yaml \
  --substitutions=_REGION=$LOCATION,_REPOSITORY=$REPOSITORY,_IMAGE_TAG=$IMAGE_TAG \
  .

gcloud builds submit \
  --config elector/cloudbuild.yaml \
  --substitutions=_REGION=$LOCATION,_REPOSITORY=$REPOSITORY,_IMAGE_TAG=$IMAGE_TAG \
  .
```

Each submission uses the Dockerfile referenced inside the `cloudbuild.yaml` file, builds the image in Cloud Build's managed environment, and automatically pushes it to Artifact Registry under the tag supplied via `_IMAGE_TAG`.

### 5. (Optional) Create pull secrets

```bash
gcloud iam service-accounts create artifact-registry-reader \
  --display-name="Artifact Registry Reader"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:artifact-registry-reader@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"
gcloud iam service-accounts keys create key.json \
  --iam-account=artifact-registry-reader@$PROJECT_ID.iam.gserviceaccount.com

kubectl create namespace churn || true
kubectl create secret docker-registry gcp-artifact-registry \
  --namespace=churn \
  --docker-server=$LOCATION-docker.pkg.dev \
  --docker-username=_json_key \
  --docker-password="$(cat key.json)" \
  --docker-email=example@example.com
rm key.json
```

Add the following to `spec.template.spec` of each deployment when using the secret:

```yaml
imagePullSecrets:
  - name: gcp-artifact-registry
```

### 6. Update Kubernetes manifests

```bash
sed -i "s|us-central1-docker.pkg.dev/PROJECT_ID/ml-models/main-model:latest|$MAIN_IMAGE|g" k8s/model-deployment.yaml
sed -i "s|us-central1-docker.pkg.dev/PROJECT_ID/ml-models/canary-model:latest|$CANARY_IMAGE|g" k8s/canary-deployment.yaml
sed -i "s|us-central1-docker.pkg.dev/PROJECT_ID/ml-models/elector:latest|$ELECTOR_IMAGE|g" k8s/elector-deployment.yaml
```

### 7. Deploy to GKE

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/model-deployment.yaml -n churn
kubectl apply -f k8s/model-svc.yaml -n churn
kubectl apply -f k8s/canary-deployment.yaml -n churn
kubectl apply -f k8s/canary-svc.yaml -n churn
kubectl apply -f k8s/elector-deployment.yaml -n churn
kubectl apply -f k8s/elector-svc.yaml -n churn
kubectl get pods -n churn
```

### 8. Expose the Elector service

```bash
kubectl patch svc elector -n churn -p '{"spec": {"type": "LoadBalancer"}}'
kubectl get svc elector -n churn
curl -X POST "http://34.58.10.230:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 29, "monthlyincome": 3800, "overtime": "Yes"}'
```

### 9. Monitor and audit

```bash
kubectl get pods -n churn
kubectl logs deployment/elector -n churn
kubectl describe pod <pod-name> -n churn
```

## Monitoring and Maintenance

- **Observability**: Export FastAPI + Elector logs to Cloud Logging or an ELK stack; add Prometheus metrics for request latency and routing ratios.
- **Model registry**: Track metric drift in `artifacts/model_registry.json` to justify increasing `CANARY_WEIGHT` or promoting/demoting models.
- **CI/CD**: Automate retraining, image builds, and GKE rollouts with Cloud Build or GitHub Actions.
- **Security**: Lock down Artifact Registry, use workload identity for GKE, and front Elector with an HTTPS ingress or API gateway.

## Cleanup

```bash
minikube stop && minikube delete
kubectl delete namespace churn || true

gcloud compute instances stop ml-deployment-vm --project=$PROJECT_ID --zone=$LOCATION-a
gcloud compute instances delete ml-deployment-vm --project=$PROJECT_ID --zone=$LOCATION-a
gcloud container clusters delete $CLUSTER_NAME --project=$PROJECT_ID --zone=$LOCATION-a
```

## Troubleshooting Common Issues

### ImagePullBackOff errors

```bash
kubectl get pods -n churn
kubectl describe pod <pod-name> -n churn
```

Typical fixes:

1. Verify manifest images reference the tags you built/pushed (e.g., `$MAIN_IMAGE`).
2. For Minikube, set `imagePullPolicy: Never` and keep tags consistent with `minikube image load`.
3. If using private Artifact Registry, confirm the `gcp-artifact-registry` secret exists in the `churn` namespace and appears under `imagePullSecrets`.

### Elector routing does not match expected weights

- Fetch `/health` from the Elector service to read the active `canary_weight`.
- Inspect logs with `kubectl logs deployment/elector -n churn` to confirm which model served each request.
- After changing `CANARY_WEIGHT`, run `kubectl rollout restart deployment/elector -n churn` so pods pick up the new environment variable.
- Ensure the canary and main services report `status: healthy`; the Elector will fall back to available models when one fails, which can skew routing percentages.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Prediction Examples (Full Payload)

The Elector service expects the complete churn feature vector. Replace `ELECTOR_HOST` with the Load Balancer IP or local host/port you are targeting.

**Class 0 (retain) example**

```bash
curl -s -X POST "http://ELECTOR_HOST/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 37,
    "businesstravel": "Travel_Rarely",
    "dailyrate": 1020,
    "department": "Research & Development",
    "distancefromhome": 6,
    "education": 3,
    "educationfield": "Medical",
    "environmentsatisfaction": 4,
    "gender": "Male",
    "hourlyrate": 68,
    "jobinvolvement": 3,
    "joblevel": 2,
    "jobrole": "Research Scientist",
    "disobediencerules": "No",
    "jobsatisfaction": 4,
    "maritalstatus": "Married",
    "monthlyincome": 5500,
    "monthlyrate": 14200,
    "numcompaniesworked": 2,
    "overtime": "No",
    "percentsalaryhike": 12,
    "performancerating": 3,
    "relationshipsatisfaction": 3,
    "stockoptionlevel": 1,
    "totalworkingyears": 11,
    "trainingtimeslastyear": 3,
    "worklifebalance": 3,
    "yearsatcompany": 8,
    "yearsincurrentrole": 5,
    "yearssincelastpromotion": 1,
    "yearswithcurrmanager": 4
  }'
```

Sample response:

```json
{
  "prediction": {
    "predicted_label": 0,
    "churn_probability": 0.05,
    "probability_vector": [0.9499, 0.0501]
  }
}
```

**Class 1 (attrite) example**

```bash
curl -s -X POST "http://ELECTOR_HOST/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 27,
    "businesstravel": "Travel_Frequently",
    "dailyrate": 350,
    "department": "Sales",
    "distancefromhome": 22,
    "education": 1,
    "educationfield": "Marketing",
    "environmentsatisfaction": 1,
    "gender": "Female",
    "hourlyrate": 42,
    "jobinvolvement": 1,
    "joblevel": 1,
    "jobrole": "Sales Executive",
    "disobediencerules": "Yes",
    "jobsatisfaction": 1,
    "maritalstatus": "Single",
    "monthlyincome": 2800,
    "monthlyrate": 2700,
    "numcompaniesworked": 5,
    "overtime": "Yes",
    "percentsalaryhike": 11,
    "performancerating": 3,
    "relationshipsatisfaction": 1,
    "stockoptionlevel": 0,
    "totalworkingyears": 4,
    "trainingtimeslastyear": 1,
    "worklifebalance": 1,
    "yearsatcompany": 2,
    "yearsincurrentrole": 1,
    "yearssincelastpromotion": 0,
    "yearswithcurrmanager": 1
  }'
```

Sample response:

```json
{
  "prediction": {
    "predicted_label": 1,
    "churn_probability": 0.776,
    "probability_vector": [0.224, 0.776]
  }
}
```
