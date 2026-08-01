class DockManager:
    """Runtime pallet staging for the maintained IJPE dock constraint."""

    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("Dock capacity cannot be negative")
        self.capacity = int(capacity)
        self.n_staged_pallets = 0

    def stage(self, quantity: int = 1) -> int:
        quantity = int(quantity)
        if self.n_staged_pallets + quantity > self.capacity:
            raise ValueError("Dock capacity exceeded")
        self.n_staged_pallets += quantity
        return self.n_staged_pallets

    def release(self, quantity: int | None) -> int:
        released = (
            self.n_staged_pallets
            if quantity is None
            else min(self.n_staged_pallets, max(0, int(quantity)))
        )
        self.n_staged_pallets -= released
        return released
