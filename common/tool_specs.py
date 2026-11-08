TOOL_SPECS = [
    {"type": "function", "function": {"name": "search_courses", "description": "根据课程编号、名称、院系或描述搜索课程", "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_course_detail", "description": "根据课程编号查询课程详情", "parameters": {"type": "object", "properties": {"course_id": {"type": "string"}}, "required": ["course_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "filter_courses", "description": "按院系和最低学分筛选课程", "parameters": {"type": "object", "properties": {"department": {"type": ["string", "null"]}, "min_credits": {"type": ["integer", "null"]}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "calculate", "description": "计算安全的四则运算表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"], "additionalProperties": False}}},
]
