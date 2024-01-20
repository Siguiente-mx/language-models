def message(role: str, content: str):
  return dict(role=role, content=content)

def system_message(content: str):
  return message("system", content)

def user_message(content: str):
  return message("user", content)

def assistant_message(content: str):
  return message("assistant", content)
