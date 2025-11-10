
// Simple function to fetch user data
async function getUser(userId) {
    const response = await fetch(`/api/users/${userId}`);
    const data = await response.json();
    return data;
}
