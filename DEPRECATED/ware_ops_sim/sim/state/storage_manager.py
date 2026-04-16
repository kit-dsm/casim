import copy

from ware_ops_algos.domain_models import Articles, StorageLocations


class StorageManager:
    def __init__(self, articles: Articles, storage: StorageLocations):
        self._articles = articles
        self._storage = storage

    def get_articles(self) -> Articles:
        return self._articles

    def get_storage(self) -> StorageLocations:
        return self._storage

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ('_storage', '_articles'):
                setattr(result, k, v)  # shared ref, never mutate
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result