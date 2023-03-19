# Kubeflow pipelines MNIST example

For further information about this repository, refer to the corresponding [Medium article]().

## Prerequisites

Tested with:

- Python: 3.8.16
- Docker: 20.10.22
- Kind 0.17.0

## How to run

### Setup Kubeflow pipelines

For information refer to the [official documentation](https://www.kubeflow.org/docs/components/pipelines/v1/installation/localcluster-deployment/).

To summarize the steps:

```bash
kind create cluster

export PIPELINE_VERSION=1.8.5
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"

# Wait a couple of minutes.
kubectl config set-context --current --namespace=kubeflow

# Port forwading for the UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Kubeflow UI will be at [http://localhost:8080](http://localhost:8080/).

### Setup venv

Local environment is set up by using [make](https://www.gnu.org/software/make/) tool.

```bash
make
```

### Build custom component

The task `train_model` involves [component specification](https://www.kubeflow.org/docs/components/pipelines/v1/sdk-v2/component-development/#creating-a-component-specification) and docker registry. For further details refer to the official documentation or the Medium article.

#### Setup docker registry


```bash
kubectl -n kubeflow create secret docker-registry registry-secret \
--docker-server=https://index.docker.io/v1/ \
--docker-username=<username> \
--docker-password=<access-key> \
--docker-email=<email>
```

#### Build and push image

```bash
source venv/bin/activate
cd custom_components/train_model_component
kfp components build . --component-filepattern train_model.py
docker tag test_kubeflow_train_model:latest <username>/kubeflow:latest
docker push <username>/kubeflow:latest
```

### Run

Make sure that [the component metadata](./custom_components/train_model_component/component_metadata/train_model.yaml) image name is correct, otherwise update it with the tag used to push the image into the Docker registry in the previous step.

```bash
source venv/bin/activate
python pipeline.py
```

### Tear down environment

```bash
kind delete cluster
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
