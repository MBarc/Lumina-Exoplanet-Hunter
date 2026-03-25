"""
Star detail endpoint.

GET /stars/{tic_id} — fetch stellar metadata from MAST + our own candidate history

MAST provides stellar parameters (temperature, radius, magnitude, etc.) via
their TIC (TESS Input Catalog) API. We enrich this with our own candidate
history so the dashboard can show everything in one request.
"""

from __future__ import annotations
import httpx
from fastapi import APIRouter, HTTPException

from api import database as db
from api.config import get_settings
from api.schemas import StarDetail, CandidateResponse

router = APIRouter(prefix="/stars", tags=["stars"])

# MAST TIC API — no authentication required, public endpoint
_MAST_TIC_URL = "https://mast.stsci.edu/api/v0/invoke"


async def _fetch_tic_metadata(tic_id: str) -> dict:
    """
    Query the MAST TIC (TESS Input Catalog) for stellar parameters.

    Uses the MAST API's Catalogs.Tic service. Returns an empty dict if
    the star isn't found or MAST is unreachable — the endpoint degrades
    gracefully so the dashboard still shows our candidate data.
    """
    payload = {
        "service": "Mast.Catalogs.Filtered.Tic",
        "format":  "json",
        "params": {
            "columns": "ID,ra,dec,Tmag,Teff,rad,mass,d",
            "filters": [{"paramName": "ID", "values": [tic_id]}],
        },
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_MAST_TIC_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("data", [])
            return rows[0] if rows else {}
    except Exception:
        # MAST is unreachable or returned an error — not fatal
        return {}


@router.get("/{tic_id}", response_model=StarDetail)
async def get_star(tic_id: str):
    """
    Return everything we know about a star.

    Combines:
      1. Stellar parameters from the MAST TIC (temperature, radius, etc.)
      2. All transit candidates our network has found for this star
      3. How many times this star has been processed
      4. A direct MAST portal URL the user can click to explore further

    The MAST portal URL uses the TIC ID as the search query so it opens
    directly to the star's page in the archive.
    """
    settings = get_settings()

    # Fetch stellar metadata and our candidate history in parallel
    import asyncio
    tic_meta, candidates_cursor = await asyncio.gather(
        _fetch_tic_metadata(tic_id),
        db.candidates().find(
            {"tic_id": tic_id},
            projection={"global_view": 0, "local_view": 0},
        ).sort("exonet_score", -1).to_list(length=50),
    )

    times_processed = await db.processed_log().count_documents({"tic_id": tic_id})

    # Build the MAST portal deep-link for this star
    import urllib.parse
    search_query = urllib.parse.quote(
        f'{{"service":"TIC","params":{{"tic_id":"{tic_id}"}}}}'
    )
    mast_url = f"{settings.mast_portal_url}?searchQuery={search_query}"

    # Serialise candidate documents
    our_candidates = []
    for doc in candidates_cursor:
        doc["id"] = str(doc.pop("_id"))
        our_candidates.append(CandidateResponse(**doc))

    return StarDetail(
        tic_id          = tic_id,
        ra              = tic_meta.get("ra"),
        dec             = tic_meta.get("dec"),
        magnitude       = tic_meta.get("Tmag"),
        effective_temp  = tic_meta.get("Teff"),
        stellar_radius  = tic_meta.get("rad"),
        stellar_mass    = tic_meta.get("mass"),
        distance_pc     = tic_meta.get("d"),
        mast_url        = mast_url,
        our_candidates  = our_candidates,
        times_processed = times_processed,
    )
