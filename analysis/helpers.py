from pydantic import BaseModel

def json_to_llm_string(data, indent=0):
    lines = []
    prefix = "  " * indent

    if isinstance(data, dict):
        for k, v in data.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}:")
                lines.append(json_to_llm_string(v, indent + 1))
            else:
                lines.append(f"{prefix}{k}: {v}")

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(json_to_llm_string(item, indent + 1))
            else:
                lines.append(f"{prefix}- {item}")

    else:
        lines.append(f"{prefix}{data}")

    return "\n".join(lines)