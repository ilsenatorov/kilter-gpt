"""
Setup DB connection.
"""

from kiltergpt.app.config import settings
from supabase import create_client

supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
