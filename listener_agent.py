# listener_agent.py

def get_mentions(product_name: str) -> list:
    """
    Simulates fetching recent mentions for a given product from social platforms.
    In a real implementation, connect to Twitter, Reddit, etc.
    """
    mock_data = [
        f"I love the design of the new {product_name}! So sleek.",
        f"The {product_name} battery life sucks. Needs charging all the time!",
        f"I'm still waiting for my {product_name} to arrive. Shipping delays are insane.",
        f"{product_name} is overpriced, but I can't lie – it's beautiful.",
        f"Why is {product_name} heating up so fast? Not safe."
    ]
    return mock_data
