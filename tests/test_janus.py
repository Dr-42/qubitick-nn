from qp.janus.tier_balance import tier_janus_loss

def test_balance():
    bal = {"a": {"writes": [1, 2, 3], "evicts": [3, 2, 1]}}; assert float(tier_janus_loss(bal)) == 0
    assert float(tier_janus_loss({"a": {"writes": [2], "evicts": [0]}})) == 4
