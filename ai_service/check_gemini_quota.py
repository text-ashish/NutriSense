import google.auth
from googleapiclient.discovery import build

# Replace with your project ID
project_id = 'nth-armor-473121-e8'

# Authenticate using your API key
credentials = google.auth.credentials.Credentials.from_authorized_user_info(info=None, api_key='AIzaSyA4MeOUoZGmbBR8FEu5A0QeK-zTsYdlEK4')

# Build the service object
service = build('serviceusage', 'v1', credentials=credentials)

# Define the quota metric for Gemini API
quota_metric = 'generativelanguage.googleapis.com/generate_content_free_tier_requests'

# Fetch the quota usage
request = service.services().get(
    name=f'projects/{project_id}/services/generativelanguage.googleapis.com'
)
response = request.execute()

# Extract and print the quota usage
for quota in response.get('quota', {}).get('usage', []):
    if quota['metric'] == quota_metric:
        print(f"Quota Metric: {quota['metric']}")
        print(f"Usage: {quota['usage']}")
        print(f"Limit: {quota['limit']}")
