import multiprocessing
import os
import time

import oci
import oci_metrics

from ads.opctl.distributed.common.cluster_runner import ClusterRunner
from ads.opctl.distributed.common.cluster_provider_factory import ClusterProviderFactory
from pytorch_cluster import PyTorchProvider


METRIC_SUBMISSION_INTERVAL_SECONDS = 60


def collect_metrics():
    if "METRICS_NAMESPACE" not in os.environ:
        return
    if "JOB_RUN_COMPARTMENT_OCID" not in os.environ:
        return
    try:
        signer = oci.auth.signers.get_resource_principals_signer()
    except EnvironmentError:
        return
    client = oci.monitoring.MonitoringClient(
        config={},
        signer=signer,
        service_endpoint=f"https://telemetry-ingestion.{signer.region}.oraclecloud.com"
    )

    while True:
        try:
            oci_metrics.submit_metrics(client)
        except:
            pass
        time.sleep(METRIC_SUBMISSION_INTERVAL_SECONDS)


if __name__ == "__main__":
    p = multiprocessing.Process(target=collect_metrics)
    p.daemon = True
    p.start()
    ClusterProviderFactory.register("PYTORCH", PyTorchProvider)
    ClusterRunner().run()
