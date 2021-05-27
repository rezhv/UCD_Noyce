import re

def remove_URL(text):
  try:
    res = re.sub(r"http\S+", "", text)
    return re.sub(r"\s*[^ /]+/[^ /]+", "", res)
  except:
    return text

def remove_hashtag(text):
  try:
    return re.sub(r"#", "", text)
  except:
    return text

def remove_username(text):
  try:
    return re.sub(r"@[^\s]+",'',text)
  except:
    return text

def normalize(text, remove_username = True, remove_hashtag= True, remove_URL = True):
  result = text
  if remove_username:
    result = remove_username(result)
  
  if remove_hashtag:
    result = remove_hashtag(result)
  
  if remove_URL:
    result = remove_URL(result)
  
  return result