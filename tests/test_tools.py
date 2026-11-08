import pytest

from common.tools import calculate, filter_courses, get_course_detail, search_courses
from from_scratch.agent import execute_tool


def test_search_and_detail():
    courses = search_courses("人工智能")
    assert courses[0]["id"] == "CS205"
    assert get_course_detail("CS101")["credits"] == 3


def test_filter_and_calculate():
    rows = filter_courses("计算机学院", 3)
    assert {row["id"] for row in rows} == {"CS101", "CS205"}
    assert calculate("3 * (2 + 1)") == 9


def test_unknown_tool_and_safe_calculator():
    with pytest.raises(ValueError):
        execute_tool("delete_files", {})
    with pytest.raises(ValueError):
        calculate("__import__('os').system('rm -rf /')")

