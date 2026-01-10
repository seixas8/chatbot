# src/calendar_service.py

import os
from datetime import datetime, timedelta
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/calendar"]

CREDENTIALS_FILE = "credentials.json"  
TOKEN_FILE = "token.json"             


def get_calendar_service():
    creds: Optional[Credentials] = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES,
            )
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds)
    return service


def create_reservation_event(
    name: str,
    people: int,
    date_str: str,
    time_str: str,
    phone: Optional[str] = None,
    notes: Optional[str] = None,
    calendar_id: str = "primary",
) -> str:

    service = get_calendar_service()

    start_dt = datetime.fromisoformat(f"{date_str}T{time_str}")
    end_dt = start_dt + timedelta(hours=2)

    description_lines = []
    if phone:
        description_lines.append(f"Telefone: {phone}")
    if notes:
        description_lines.append(f"Notas: {notes}")

    description = "\n".join(description_lines)

    event = {
        "summary": f"Reserva {people}p - {name}",
        "description": description,
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": "Europe/Lisbon",
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": "Europe/Lisbon",
        },
    }

    created_event = service.events().insert(
        calendarId=calendar_id,
        body=event,
    ).execute()

    return created_event.get("id", "")
