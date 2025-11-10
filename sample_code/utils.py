
def sort_list_ascending(my_list):
    """Sorts a given list in ascending order."""
    return sorted(my_list)

def calculate_average(numbers):
    """Calculates the mean of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
