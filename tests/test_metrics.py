import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)
from src.load_data import clean_employee_data


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_df(**overrides):
    """Minimal valid DataFrame; override any column as needed."""
    base = {
        "employee_id": [1, 2, 3, 4],
        "department": ["Sales", "Sales", "HR", "HR"],
        "overtime": ["Yes", "No", "No", "Yes"],
        "travel_frequency": ["Frequent", "Rarely", "Occasional", "Frequent"],
        "job_satisfaction": [1, 3, 4, 2],
        "monthly_income": [4000, 6000, 7000, 5000],
        "years_at_company": [2, 5, 8, 3],
        "age": [28, 35, 42, 30],
        "attrition": ["Yes", "No", "No", "Yes"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# attrition_rate
# ---------------------------------------------------------------------------

def test_attrition_rate_returns_expected_percent():
    df = make_df(attrition=["Yes", "No", "No", "Yes"])
    assert attrition_rate(df) == 50.0


def test_attrition_rate_zero_when_no_leavers():
    df = make_df(attrition=["No", "No", "No", "No"])
    assert attrition_rate(df) == 0.0


def test_attrition_rate_100_when_all_leave():
    df = make_df(attrition=["Yes", "Yes", "Yes", "Yes"])
    assert attrition_rate(df) == 100.0


# ---------------------------------------------------------------------------
# attrition_by_department
# ---------------------------------------------------------------------------

def test_attrition_by_department_returns_expected_columns():
    result = attrition_by_department(make_df())
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_calculates_correct_rates():
    # Sales: 1 leaver out of 2 = 50%; HR: 1 leaver out of 2 = 50%
    result = attrition_by_department(make_df())
    rates = dict(zip(result["department"], result["attrition_rate"]))
    assert rates["Sales"] == 50.0
    assert rates["HR"] == 50.0


def test_attrition_by_department_sorted_descending():
    # Operations: 2/2 = 100%, Sales: 0/2 = 0%
    df = make_df(
        employee_id=[1, 2, 3, 4],
        department=["Sales", "Sales", "Operations", "Operations"],
        attrition=["No", "No", "Yes", "Yes"],
    )
    result = attrition_by_department(df)
    assert list(result["department"]) == ["Operations", "Sales"]


def test_attrition_by_department_leaver_counts():
    result = attrition_by_department(make_df())
    sales_row = result[result["department"] == "Sales"].iloc[0]
    assert sales_row["employees"] == 2
    assert sales_row["leavers"] == 1


# ---------------------------------------------------------------------------
# attrition_by_overtime
# ---------------------------------------------------------------------------

def test_attrition_by_overtime_returns_expected_columns():
    result = attrition_by_overtime(make_df())
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_calculates_correct_rates():
    # Yes group (employees 1, 4): both left → 100%
    # No group (employees 2, 3): none left → 0%
    result = attrition_by_overtime(make_df())
    rates = dict(zip(result["overtime"], result["attrition_rate"]))
    assert rates["Yes"] == 100.0
    assert rates["No"] == 0.0


def test_attrition_by_overtime_leaver_counts():
    result = attrition_by_overtime(make_df())
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    assert yes_row["employees"] == 2
    assert yes_row["leavers"] == 2


# ---------------------------------------------------------------------------
# average_income_by_attrition
# ---------------------------------------------------------------------------

def test_average_income_by_attrition_returns_expected_columns():
    result = average_income_by_attrition(make_df())
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_calculates_correct_means():
    # Leavers (Yes): employees 1 (4000) and 4 (5000) → mean = 4500
    # Stayers (No): employees 2 (6000) and 3 (7000) → mean = 6500
    result = average_income_by_attrition(make_df())
    means = dict(zip(result["attrition"], result["avg_monthly_income"]))
    assert means["Yes"] == 4500.0
    assert means["No"] == 6500.0


# ---------------------------------------------------------------------------
# satisfaction_summary
# ---------------------------------------------------------------------------

def test_satisfaction_summary_returns_expected_columns():
    result = satisfaction_summary(make_df())
    assert "job_satisfaction" in result.columns
    assert "total_employees" in result.columns
    assert "leavers" in result.columns
    assert "attrition_rate" in result.columns


def test_satisfaction_summary_divides_by_group_size_not_total_leavers():
    # Satisfaction 1: 1 employee, 1 leaver → rate should be 100%, not 50%
    # Satisfaction 3: 1 employee, 0 leavers → rate should be 0%
    # If the old bug were present (dividing by total leavers=2):
    #   sat=1 would be 1/2*100 = 50%, not 100%
    df = make_df(
        employee_id=[1, 2, 3, 4],
        job_satisfaction=[1, 3, 4, 2],
        attrition=["Yes", "No", "No", "Yes"],
    )
    result = satisfaction_summary(df)
    rates = dict(zip(result["job_satisfaction"], result["attrition_rate"]))
    assert rates[1] == 100.0
    assert rates[3] == 0.0


def test_satisfaction_summary_sorted_ascending():
    result = satisfaction_summary(make_df())
    sat_values = list(result["job_satisfaction"])
    assert sat_values == sorted(sat_values)


def test_satisfaction_summary_leaver_counts_are_correct():
    # Satisfaction 1: employee 1 left; satisfaction 2: employee 4 left
    result = satisfaction_summary(make_df())
    counts = dict(zip(result["job_satisfaction"], result["leavers"]))
    assert counts[1] == 1
    assert counts[2] == 1
    assert counts[3] == 0
    assert counts[4] == 0


# ---------------------------------------------------------------------------
# clean_employee_data
# ---------------------------------------------------------------------------

def test_clean_raises_on_missing_column():
    df = make_df().drop(columns=["attrition"])
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_employee_data(df)


def test_clean_fills_missing_department():
    df = make_df()
    df.loc[0, "department"] = None
    result = clean_employee_data(df)
    assert result.loc[0, "department"] == "Unknown"


def test_clean_fills_missing_overtime():
    df = make_df()
    df.loc[0, "overtime"] = None
    result = clean_employee_data(df)
    assert result.loc[0, "overtime"] == "No"


def test_clean_fills_missing_job_satisfaction():
    df = make_df()
    df.loc[0, "job_satisfaction"] = None
    result = clean_employee_data(df)
    assert result.loc[0, "job_satisfaction"] == 3


def test_clean_title_cases_attrition():
    df = make_df(attrition=["yes", "NO", "Yes", "no"])
    result = clean_employee_data(df)
    assert list(result["attrition"]) == ["Yes", "No", "Yes", "No"]


def test_clean_strips_whitespace_from_department():
    df = make_df(department=["  Sales", "HR  ", " IT ", "Finance"])
    result = clean_employee_data(df)
    assert list(result["department"]) == ["Sales", "HR", "IT", "Finance"]


def test_clean_does_not_mutate_original():
    df = make_df()
    df.loc[0, "department"] = None
    clean_employee_data(df)
    assert pd.isna(df.loc[0, "department"])
