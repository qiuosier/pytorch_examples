# Training minGPT on Oracle Cloud Infrastructure using Oracle ADS

Here are the steps to train the minGPT model on Oracle Cloud Infrastructure using Oracle ADS:

1. Configure your access to OCI Data Science, Container Registry and Object Storage.
2. Build the container image.
    ```
    export IMAGE_NAME=<region.ocir.io/my-tenancy/image-name>
    export TAG=latest
    ads opctl distributed-training build-image \
        -t $TAG \
        -reg $IMAGE_NAME \
        -df oci_dist_training_artifacts/pytorch/v1_metrics/Dockerfile
    ```
3. Run the training. This will also push the image to container registry.
    ```
    ads opctl run -f pytorch_mingpt_fsdp.yaml
    ```

See:
* [ADS PyTorch Distributed Training Documentation](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/distributed_training/pytorch/pytorch.html)
