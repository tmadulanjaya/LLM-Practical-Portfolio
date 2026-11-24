"""
PUSL 3205 - Lab 6, Part D: Google Calendar tools

Install:
    pip install google-api-python-client google-auth-oauthlib google-auth-httplib2

Needs credentials.json (OAuth Desktop client) in the same folder.
The first run opens a browser to sign in; it then saves token.json.
"""
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.tools import tool

SCOPES = ["https://www.googleapis.com/auth/calendar"]
TZ = ZoneInfo("Asia/Colombo")
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"


def _get_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def _parse_date(date: str):
    d = (date or "today").strip().lower()
    today = datetime.now(TZ).date()
    if d in ("today", ""):
        return today
    if d == "tomorrow":
        return today + timedelta(days=1)
    return datetime.strptime(d, "%Y-%m-%d").date()


@tool
def check_schedule(date: str = "today") -> str:
    """Check the Google Calendar schedule for a day.
    date can be 'today', 'tomorrow' or a date like '2026-11-15'."""
    try:
        day = _parse_date(date)
    except ValueError:
        return "Invalid date. Use 'today', 'tomorrow' or YYYY-MM-DD."
    start = datetime(day.year, day.month, day.day, tzinfo=TZ)
    end = start + timedelta(days=1)
    events = _get_service().events().list(
        calendarId="primary",
        timeMin=start.isoformat(), timeMax=end.isoformat(),
        singleEvents=True, orderBy="startTime",
    ).execute().get("items", [])

    if not events:
        return f"No events on {day}."
    lines = []
    for e in events:
        s = e["start"].get("dateTime", e["start"].get("date"))
        if "T" in s:
            s = datetime.fromisoformat(s).astimezone(TZ).strftime("%I:%M %p")
        else:
            s = "All day"
        lines.append(f"{s} - {e.get('summary', '(no title)')}")
    return f"Events on {day}:\n" + "\n".join(lines)


@tool
def create_calendar_event(title: str, date: str, start_time: str,
                          duration_minutes: int = 60) -> str:
    """Create a new Google Calendar event.
    title: event name. date: 'today', 'tomorrow' or YYYY-MM-DD.
    start_time: 24-hour 'HH:MM', e.g. '14:30'. duration_minutes: length."""
    try:
        day = _parse_date(date)
        hh, mm = start_time.strip().split(":")
        start = datetime(day.year, day.month, day.day, int(hh), int(mm), tzinfo=TZ)
    except ValueError:
        return "Invalid date or time. Use date YYYY-MM-DD and start_time HH:MM."
    end = start + timedelta(minutes=int(duration_minutes))
    event = _get_service().events().insert(
        calendarId="primary",
        body={
            "summary": title,
            "start": {"dateTime": start.isoformat(), "timeZone": "Asia/Colombo"},
            "end": {"dateTime": end.isoformat(), "timeZone": "Asia/Colombo"},
        },
    ).execute()
    return (f"Event created: '{title}' on {day} at "
            f"{start.strftime('%I:%M %p')} ({duration_minutes} min). "
            f"Link: {event.get('htmlLink')}")


if __name__ == "__main__":
    # Quick test: signs you in on first run, then prints today's events
    print(check_schedule.invoke({"date": "today"}))
