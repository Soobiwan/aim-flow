from aim_flow.schedules import get_schedule_weight


def test_schedule_values_are_bounded() -> None:
    schedules = ["constant", "early", "middle", "late", "early_middle", "middle_late"]
    for schedule in schedules:
        for step in range(12):
            value = get_schedule_weight(schedule, step, 12)
            assert 0.0 <= value <= 1.0


def test_early_starts_higher_than_it_ends() -> None:
    assert get_schedule_weight("early", 0, 20) > get_schedule_weight("early", 19, 20)


def test_late_ends_higher_than_it_starts() -> None:
    assert get_schedule_weight("late", 19, 20) > get_schedule_weight("late", 0, 20)


def test_middle_peaks_near_center() -> None:
    center = get_schedule_weight("middle", 10, 21)
    start = get_schedule_weight("middle", 0, 21)
    end = get_schedule_weight("middle", 20, 21)
    assert center > start
    assert center > end

