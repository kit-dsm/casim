from ware_ops_algos.domain_models import Articles, StorageLocations


class StorageManager:
    def __init__(self, articles: Articles, storage: StorageLocations):
        self._articles = articles
        self._storage = storage

    def get_articles(self) -> Articles:
        return self._articles

    def get_storage(self) -> StorageLocations:
        return self._storage
