import kfp

from kfp.v2 import dsl
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Artifact,
    Dataset
)
import argparse
from kubernetes import client as k8s_client


download_link = 'https://github.com/kubeflow/examples/blob/master/digit-recognition-kaggle-competition/data/{file}.csv.zip?raw=true'


@component(
    packages_to_install=["wget", "pandas"],
    base_image="python:3.8"
)
def download_data(download_link: str, train: Output[Dataset], test: Output[Dataset]):
    import zipfile
    import wget
    import logging
    import pandas as pd

    data_path = '/tmp'

    # download files
    wget.download(download_link.format(file='train'),
                  f'{data_path}/train_csv.zip')
    wget.download(download_link.format(file='test'),
                  f'{data_path}/test_csv.zip')
    logging.info("Download completed.")

    with zipfile.ZipFile(f"{data_path}/train_csv.zip", "r") as zip_ref:
        zip_ref.extractall(data_path)

    with zipfile.ZipFile(f"{data_path}/test_csv.zip", "r") as zip_ref:
        zip_ref.extractall(data_path)

    logging.info('Extraction completed, path is %s', data_path)

    train_df = pd.read_csv(
        f"{data_path}/train.csv").to_csv(train.path, index=False)
    test_df = pd.read_csv(
        f"{data_path}/test.csv").to_csv(test.path, index=False)


@component(
    packages_to_install=["pandas", "scikit-learn", "torch"],
    base_image="python:3.8"
)
def pre_process_data(train: Input[Dataset], test: Input[Dataset], train_tensor_path: Output[Artifact],
                     val_tensor_path: Output[Artifact], test_tensor_path: Output[Artifact]):
    import torch
    import pandas as pd

    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset

    train_df = pd.read_csv(filepath_or_buffer=train.path)
    test_df = pd.read_csv(filepath_or_buffer=test.path)

    train_labels = train_df['label'].values
    train_images = (train_df.iloc[:, 1:].values).astype('float32')
    test_images = (test_df.iloc[:, :].values).astype('float32')

    # Training and Validation Split
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,
                                                                          stratify=train_labels, random_state=123,
                                                                          test_size=0.20)

    train_images = train_images.reshape(train_images.shape[0], 28, 28)
    val_images = val_images.reshape(val_images.shape[0], 28, 28)
    test_images = test_images.reshape(test_images.shape[0], 28, 28)

    # train
    train_images_tensor = torch.tensor(train_images)/255.0
    train_labels_tensor = torch.tensor(train_labels)
    train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)
    torch.save(train_tensor, train_tensor_path.path)

    # val
    val_images_tensor = torch.tensor(val_images)/255.0
    val_labels_tensor = torch.tensor(val_labels)
    val_tensor = TensorDataset(val_images_tensor, val_labels_tensor)
    torch.save(val_tensor, val_tensor_path.path)

    # test
    test_tensor = torch.tensor(test_images)/255.0
    torch.save(test_tensor, test_tensor_path.path)


@dsl.pipeline(name="digit-recognizer-pipeline",
              description="Performs Preprocessing, training and prediction of digits")
def digit_recognize_pipeline(download_link: str
                             ):

    # Create download container.
    generate_datasets = download_data(download_link)
    preprocess_tensors = pre_process_data(
        generate_datasets.outputs['train'], generate_datasets.outputs['test'])

    train_model = kfp.components.load_component_from_file(
        './custom_components/train_model_component/component_metadata/train_model.yaml')

    model = train_model(preprocess_tensors.outputs['train_tensor_path'],
                        preprocess_tensors.outputs['val_tensor_path'],
                        preprocess_tensors.outputs['test_tensor_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MNIST Kubeflow example")
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--no-run', dest='run', action='store_false')
    parser.set_defaults(run=True)

    args = parser.parse_args()

    # create client that would enable communication with the Pipelines API server
    client = kfp.Client()

    arguments = {"download_link": download_link}

    pipeline_conf = kfp.dsl.PipelineConf()
    pipeline_conf.set_image_pull_secrets(
        [k8s_client.V1ObjectReference(name="registry-secret")])

    if args.run == 1:
        client.create_run_from_pipeline_func(digit_recognize_pipeline, arguments=arguments,
                                             experiment_name="mnist",
                                             pipeline_conf=pipeline_conf,
                                             mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE)
    else:
        kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
            pipeline_func=digit_recognize_pipeline, package_path='output_mnist.yaml')
        client.upload_pipeline_version(pipeline_package_path='output_mnist.yaml', pipeline_version_name="0.3",
                                       pipeline_name="MNIST example pipeline", description="Example pipeline")
