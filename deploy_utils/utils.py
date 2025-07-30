def reindex(from_order: list[str], to_order: list[str]):
    """
    Reindex the order of the list from the from_order to the to_order.
    """
    return [from_order.index(item) for item in to_order]

