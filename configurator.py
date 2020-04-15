from dwave.cloud import Client


def configure_cloud_system():
    client = Client.from_config()
    client.get_solvers()
