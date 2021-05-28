import json
import os
import shutil
from io import StringIO
from typing import List

import pandas as pd
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient

from .logging_helper import LoggingHelper


class BlobStorageHelper(object):
    LOGGER = LoggingHelper("BlobStorageHelper").logger

    def __init__(self, account_name: str = None, connection_string: str = None):
        self.storage_account_name = account_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.local_base_path = "tmp/"

    def create_container(self, container_name: str):
        try:
            self.blob_service_client.create_container(name=container_name)
            BlobStorageHelper.LOGGER.info(
                "Container [{}] created".format(container_name)
            )
        except ResourceExistsError:
            BlobStorageHelper.LOGGER.info(
                "Container [{}] already exists".format(container_name)
            )

    def get_container_client(self, container_name: str):
        return self.blob_service_client.get_container_client(container_name)

    def get_blob_client(self, container_name: str, blob_name: str):
        return self.blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

    def create_local_dir(self, dir_path: str):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

    def clean_local_folder(self):
        shutil.rmtree(self.local_base_path)

    def get_file_as_bytes(self, container_name: str, remote_file_name: str) -> bytes:
        blob_client = self.get_blob_client(container_name, remote_file_name)
        return blob_client.download_blob().readall()

    def get_file_as_text(self, container_name: str, remote_file_name: str) -> str:
        return self.get_file_as_bytes(container_name, remote_file_name).decode("UTF-8")

    def get_list_blobs_name(self, container_name: str, prefix: str = None) -> List[str]:
        container_client = self.get_container_client(container_name)
        list_blobs = []
        if prefix is not None:
            for blob in container_client.list_blobs(name_starts_with=prefix):
                list_blobs.append(blob.name)
        else:
            for blob in container_client.list_blobs():
                list_blobs.append(blob.name)
        return list_blobs

    def download_file_locally(
            self, container_name: str, remote_file_name: str, local_file_name: str = None
    ):
        BlobStorageHelper.LOGGER.info(
            "Downloading file [{}] from container [{}]".format(
                remote_file_name, container_name
            )
        )
        if local_file_name is None:
            self.create_local_dir(self.local_base_path)
            local_file_name = self.local_base_path + remote_file_name
        else:
            local_file_name = local_file_name

        blob_client = self.get_blob_client(container_name, remote_file_name)

        with open(local_file_name, "wb") as my_blob:
            my_blob.write(blob_client.download_blob().readall())
        BlobStorageHelper.LOGGER.info(
            "File downloaded here [{}]".format(local_file_name)
        )

    def upload_file_to_blob(
            self, container_name: str, local_file_name: str, remote_file_name: str
    ):
        self.create_container(container_name)
        container_client = self.get_container_client(container_name)
        blob_client = container_client.get_blob_client(remote_file_name)
        with open(local_file_name, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    def upload_bytes_to_blob(
            self, bytes: bytes, container_name: str, remote_file_name: str
    ):
        self.create_container(container_name)
        container_client = self.get_container_client(container_name)
        blob_client = container_client.get_blob_client(remote_file_name)
        blob_client.upload_blob(bytes, overwrite=True)

    def get_csv_as_df(self, container_name: str, remote_file_name: str) -> pd.DataFrame:
        return pd.read_csv(
            StringIO(self.get_file_as_text(container_name, remote_file_name))
        )

    def get_file_as_json(self, container_name: str, remote_file_name: str):
        return json.loads(self.get_file_as_bytes(container_name, remote_file_name))
