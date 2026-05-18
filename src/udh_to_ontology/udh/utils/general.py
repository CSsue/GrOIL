

def batch_list(items, max_batch_size=40):
    """Batch a list into chunks of max_batch_size."""
    return [items[i:i + max_batch_size] for i in range(0, len(items), max_batch_size)]