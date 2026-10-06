"""
Setup DB connection.
"""

from kiltergpt.app.config import settings
from supabase import create_client


def get_supabase_client():
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
