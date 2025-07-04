
def combineListsRemoveDuplicates(list1, list2):
  """combine two lists and remove duplicates

  Args:
      list1 (list): the first list of words
      list2 (list): the second list of words

  Returns:
      list: the combined list of words
  """
  combinedList = list1 + list2
  seen = set()
  result = []

  for item in combinedList:
      if item not in seen:
          seen.add(item)
          result.append(item)

  return result
