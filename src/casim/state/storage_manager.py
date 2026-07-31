import copy
from collections import defaultdict

from ware_ops_algos.algorithms import PickPosition
from ware_ops_algos.domain_models import (
    Articles,
    Location,
    StorageLocations,
)


class StorageManager:
    def __init__(self, articles: Articles, storage: StorageLocations):
        self._articles = articles
        self._storage_type = storage.tpe
        self._on_hand: dict[
            tuple[int, int | float, int | float], float
        ] = defaultdict(float)
        for location in storage.locations or []:
            key = (location.article_id, location.x, location.y)
            self._on_hand[key] += float(location.amount)
        self._reservations: dict[
            int,
            dict[tuple[int, int | float, int | float], float],
        ] = {}

    def get_articles(self) -> Articles:
        return self._articles

    def get_storage(self) -> StorageLocations:
        return self.planning_snapshot()

    def planning_snapshot(self) -> StorageLocations:
        reserved = self._reserved_totals()
        snapshot = StorageLocations(
            self._storage_type,
            [
                Location(
                    x=key[1],
                    y=key[2],
                    article_id=key[0],
                    amount=self._on_hand[key] - reserved.get(key, 0.0),
                )
                for key in sorted(self._on_hand)
            ],
        )
        snapshot.build_article_location_mapping()
        return snapshot

    @staticmethod
    def _pick_key(
        pick: PickPosition,
    ) -> tuple[int, int | float, int | float]:
        return (pick.article_id, pick.pick_node[0], pick.pick_node[1])

    @staticmethod
    def _pick_quantities(
        picks: list[PickPosition] | tuple[PickPosition, ...],
    ) -> dict[tuple[int, int | float, int | float], float]:
        quantities: dict[
            tuple[int, int | float, int | float], float
        ] = defaultdict(float)
        for pick in picks:
            quantities[StorageManager._pick_key(pick)] += float(
                pick.in_store
            )
        return dict(quantities)

    def _reserved_totals(
        self,
    ) -> dict[tuple[int, int | float, int | float], float]:
        totals: dict[
            tuple[int, int | float, int | float], float
        ] = defaultdict(float)
        for reservation in self._reservations.values():
            for key, quantity in reservation.items():
                totals[key] += quantity
        return dict(totals)

    def _reserve(
        self,
        tour_id: int,
        picks: list[PickPosition] | tuple[PickPosition, ...],
        *,
        extend: bool,
    ) -> None:
        if not extend and tour_id in self._reservations:
            raise ValueError(f"Tour {tour_id} already has a reservation")
        requested = self._pick_quantities(picks)
        if not requested:
            return
        reserved = self._reserved_totals()
        for key, quantity in requested.items():
            available = self._on_hand.get(key, 0.0) - reserved.get(key, 0.0)
            if quantity > available:
                raise ValueError(
                    "Insufficient available inventory for "
                    f"article={key[0]} node={(key[1], key[2])}: "
                    f"requested={quantity}, available={available}"
                )
        reservation = self._reservations.setdefault(tour_id, {})
        for key, quantity in requested.items():
            reservation[key] = reservation.get(key, 0.0) + quantity

    def reserve(
        self,
        tour_id: int,
        picks: list[PickPosition] | tuple[PickPosition, ...],
    ) -> None:
        self._reserve(tour_id, picks, extend=False)

    def extend_reservation(
        self,
        tour_id: int,
        picks: list[PickPosition] | tuple[PickPosition, ...],
    ) -> None:
        self._reserve(tour_id, picks, extend=True)

    def confirm_pick(self, tour_id: int, pick: PickPosition) -> None:
        key = self._pick_key(pick)
        quantity = float(pick.in_store)
        reservation = self._reservations.get(tour_id)
        if reservation is None or reservation.get(key, 0.0) < quantity:
            raise ValueError(
                f"Pick {pick} is not reserved for tour {tour_id}"
            )
        if self._on_hand.get(key, 0.0) < quantity:
            raise ValueError(f"Inventory underflow for {key}")
        reservation[key] -= quantity
        self._on_hand[key] -= quantity
        if reservation[key] == 0:
            del reservation[key]
        if not reservation:
            del self._reservations[tour_id]

    def release_reservation(self, tour_id: int) -> None:
        self._reservations.pop(tour_id, None)

    def assert_reservation_empty(self, tour_id: int) -> None:
        remaining = self._reservations.get(tour_id, {})
        if remaining:
            raise ValueError(
                f"Tour {tour_id} still has unpicked inventory: {remaining}"
            )

    def adjust_on_hand(
        self,
        article_id: int,
        node: tuple[int | float, int | float],
        quantity: float,
    ) -> None:
        key = (article_id, node[0], node[1])
        updated = self._on_hand.get(key, 0.0) + float(quantity)
        reserved = self._reserved_totals().get(key, 0.0)
        if updated < reserved:
            raise ValueError(
                "Inventory adjustment would reduce on-hand below reserved "
                f"quantity for {key}: on_hand={updated}, reserved={reserved}"
            )
        self._on_hand[key] = updated

    def reservation_for(
        self,
        tour_id: int,
    ) -> dict[tuple[int, int | float, int | float], float]:
        return dict(self._reservations.get(tour_id, {}))

    def reservation_tour_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._reservations))

    def restore_reservation(
        self,
        tour_id: int,
        reservation: dict[
            tuple[int, int | float, int | float],
            float,
        ],
    ) -> None:
        """Restore a reservation captured before an atomic state operation."""
        if reservation:
            self._reservations[tour_id] = dict(reservation)
        else:
            self._reservations.pop(tour_id, None)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_articles":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result
