class User:
    """
    A simple class to represent a user in the system.
    """

    def __init__(self, username, email):
        self.username = username
        self.email = email

    def get_profile_info(self):
        """Returns a formatted string of the user's profile."""
        return f"User: {self.username}, Email: {self.email}"
