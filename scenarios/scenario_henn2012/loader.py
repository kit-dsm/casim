"""Henn generator loader for the L2 LONG selector's card requirement."""

from scenarios.scenario_oobp.loader import OobpHennLoader


class Henn2012LongLoader(OobpHennLoader):
    """Give every order an unused synthetic end-of-shift due date.

    The shared LPT card requires ``due_date`` for applicability, though its
    selection calculation sorts only by processing time. Henn publishes no
    due dates for L2. This field makes the metadata true without affecting
    the paper's LONG selection rule.
    """

    def _build(self, setting, orders, arrivals):
        domain = super()._build(setting, orders, arrivals)
        for order in domain.orders.orders:
            order.due_date = 8 * 60 * 60
        return domain
