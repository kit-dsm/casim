import numpy as np

def generate_orders(max_orders: int,
                    max_articles: int,
                    n_articles: int,
                    min_lines: int,
                    max_lines: int,
                    min_amount: int,
                    max_amount: int):
    rng = np.random.default_rng(0)
    # orders
    # "64": [{"article_id": 109, "amount": 1}, {"article_id": 116, "amount": 1}]
    n_orders = rng.integers(1, max_orders)
    orders = {}
    for o_id in range(n_orders):
        n_lines = rng.integers(low=min_lines, high=max_lines + 1)

        article_ids = rng.choice(range(n_articles), size=n_lines, replace=False)
        amounts = rng.integers(min_amount, max_amount + 1, size=n_lines)
        orders[str(o_id)] = [{"article_id": int(a_id), "amount": int(amt)} for a_id, amt in zip(article_ids, amounts)]
    return orders



def generate_arrival_times(n_orders: int, start: float, end: float):
    order_arrival_times_np = np.random.uniform(start, end, n_orders)

    order_arrival_times = {}
    for o_id in range(n_orders):
        order_arrival_times[str(o_id)] = float(order_arrival_times_np[o_id])

    return order_arrival_times

orders = generate_orders(max_orders=10,
                         max_articles=4,
                         n_articles=10,
                         min_lines=5,
                         max_lines=10,
                         min_amount=1,
                         max_amount=5)
order_arrival_times = generate_arrival_times(len(orders), 0, 1000)
print(orders)
print(order_arrival_times)

n_aisles = 6
n_pick_locations = 12

storage = {}
for article in articles:
    storage[str(article)] = {}


# "storage": [
#     {"article_id": 101, "location": [1, 2], "quantity": 8},
#     {"article_id": 102, "location": [2, 4], "quantity": 8},
#     {"article_id": 103, "location": [3, 7], "quantity": 8},
#     {"article_id": 104, "location": [4, 10], "quantity": 8},
#     {"article_id": 105, "location": [5, 15], "quantity": 8},
#     {"article_id": 106, "location": [6, 18], "quantity": 8},
#     {"article_id": 107, "location": [1, 22], "quantity": 8},
#     {"article_id": 108, "location": [2, 25], "quantity": 8},
#     {"article_id": 109, "location": [3, 3], "quantity": 8},
#     {"article_id": 110, "location": [4, 6], "quantity": 8},
#     {"article_id": 111, "location": [5, 9], "quantity": 8},
#     {"article_id": 112, "location": [6, 12], "quantity": 8},
#     {"article_id": 113, "location": [1, 16], "quantity": 8},
#     {"article_id": 114, "location": [2, 20], "quantity": 8},
#     {"article_id": 115, "location": [3, 24], "quantity": 8},
#     {"article_id": 116, "location": [4, 2], "quantity": 8},
#     {"article_id": 117, "location": [5, 5], "quantity": 8},
#     {"article_id": 118, "location": [6, 8], "quantity": 8},
#     {"article_id": 119, "location": [1, 11], "quantity": 8},
#     {"article_id": 120, "location": [2, 17], "quantity": 8},
#     {"article_id": 121, "location": [3, 21], "quantity": 8},
#     {"article_id": 122, "location": [4, 25], "quantity": 8},
#     {"article_id": 123, "location": [5, 14], "quantity": 8},
#     {"article_id": 124, "location": [6, 23], "quantity": 8}
#   ],
{
    "source": "generator",
    "release_times": order_arrival_times,
    "orders": orders,
    "n_pickers": 8,
    "batch_capacity_orders": 3,
    "pick_cart": {
        "n_boxes": 3,
        "capacity_per_bin": 1,
        "box_can_mix_orders": false
    },
    "picker_speed": 1,
    "time_per_pick": 4,
    "layout": {
        "n_aisles": 6,
        "n_pick_locations": 12,
        "n_blocks": 2,
        "dist_aisle": 3,
        "dist_pick_locations": 2,
        "dist_aisle_location": 1,
        "dist_between_blocks": 4,
        "dist_start": 0,
        "dist_end": 0,
        "start_location": [0, 0],
        "end_location": [-1, 0],
        "start_connection_point": [1, 0],
        "end_connection_point": [1, 0]
    }
}
