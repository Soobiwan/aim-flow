from aim_flow.primitive_flow import parse_aggregation_steps


def test_final_only_returns_last_step():
    assert parse_aggregation_steps(24, final_only=True) == {23}


def test_explicit_aggregation_steps_are_used():
    assert parse_aggregation_steps(24, aggregation_steps=[12, 16, 20, 23]) == {12, 16, 20, 23}


def test_fractions_convert_for_24_steps():
    assert parse_aggregation_steps(24, aggregation_step_fractions=[0.5, 0.7, 0.9, 1.0]) == {12, 16, 21, 23}


def test_aggregate_every_n_steps_includes_final():
    assert parse_aggregation_steps(10, aggregate_every_n_steps=4) == {0, 4, 8, 9}


def test_invalid_steps_are_removed():
    assert parse_aggregation_steps(5, aggregation_steps=[-1, 0, 4, 5, 99]) == {0, 4}
