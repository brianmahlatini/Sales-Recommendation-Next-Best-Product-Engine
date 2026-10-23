from app.ml.features import FEATURE_COLUMNS, build_feature_frame, generate_synthetic_retail_data, split_xy


def test_feature_frame_has_supervised_target_and_features():
    retail_data = generate_synthetic_retail_data(n_customers=24, n_products=16, seed=7)
    frame = build_feature_frame(retail_data)
    x, y = split_xy(frame)

    assert set(FEATURE_COLUMNS).issubset(x.columns)
    assert "target_purchase" in frame.columns
    assert y.isin([0, 1]).all()
    assert y.sum() > 0
    assert len(frame) == 24 * 16
