"""Build a flattened algorithm data card from a composed configuration."""

from ware_ops_algos.domain_models import DataCard, information_card_section


def load_and_flatten_data_card(raw) -> DataCard:
    def flatten_domain(domain: dict) -> dict:
        features = {}
        for obj in domain.get("objects", []):
            for feat in obj.get("features", []):
                features[feat["name"]] = feat.get("value", True)
            features.update(flatten_domain(obj))
        return features

    def section(domain: dict) -> dict:
        return {"type": domain.get("type"),
                "features": flatten_domain(domain)}

    return DataCard(
        name=raw.get("name", ""),
        problem_class=raw.get("problem_class", ""),
        objective=raw.get("objective", ""),
        layout=section(raw.get("layout", {})),
        articles=section(raw.get("articles", {})),
        orders=section(raw.get("orders", {})),
        resources=section(raw.get("resources", {})),
        storage=section(raw.get("storage", {})),
        warehouse_info=section(raw.get("warehouse_info", {})),
        information=information_card_section(raw.get("information")),
    )
