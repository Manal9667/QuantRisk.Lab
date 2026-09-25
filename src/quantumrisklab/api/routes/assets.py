"""Asset endpoint: GET /assets."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from quantumrisklab.api.schemas import AssetResponse
from quantumrisklab.db.connection import get_session
from quantumrisklab.db.repository import list_assets

router = APIRouter(tags=["assets"])


@router.get("/assets", response_model=list[AssetResponse])
def get_assets(session: Session = Depends(get_session)) -> list[AssetResponse]:
    return [
        AssetResponse(
            asset_id=a.asset_id,
            ticker=a.ticker,
            name=a.name,
            sector=a.sector,
            asset_class=a.asset_class,
            currency=a.currency,
        )
        for a in list_assets(session)
    ]
