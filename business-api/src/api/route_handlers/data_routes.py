"""Data discovery routes for Business API."""

import asyncio
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..auth import get_current_user, CurrentUser, check_rate_limit
from ..audit import audit_logger
from ..exceptions import ExternalDependencyError

router = APIRouter()


class DatasetSearchRequest(BaseModel):
    """Dataset search request."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, description="Maximum results")
    sources: Optional[List[str]] = Field(None, description="Data sources to search")
    min_images: Optional[int] = Field(None, description="Minimum images")
    license: Optional[str] = Field(None, description="License filter")


class DatasetSearchResponse(BaseModel):
    """Dataset search response."""
    datasets: List[dict]
    total: int
    query_time_ms: int = 0


@router.post("/search", response_model=DatasetSearchResponse)
async def search_datasets(
    request: DatasetSearchRequest,
    http_request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Search for datasets across multiple sources.

    Supported sources: Roboflow, Kaggle, HuggingFace

    Requires authentication.
    """
    import time
    start_time = time.time()

    try:
        from src.data.discovery import DatasetDiscovery
        discovery = DatasetDiscovery()
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, discovery.search, request.query, request.max_results
        )

        if request.sources:
            results = [r for r in results if r.source in request.sources]

        datasets = [
            {
                "name": ds.name,
                "source": ds.source,
                "url": ds.url,
                "license": ds.license,
                "images": ds.images,
                "relevance_score": ds.relevance_score,
            }
            for ds in results
        ]

        query_time_ms = int((time.time() - start_time) * 1000)

        audit_logger.log_data_access(
            user_id=current_user.user_id,
            dataset_id=request.query,
            action="search",
            request=http_request,
            details={"query": request.query, "max_results": request.max_results, "sources": request.sources}
        )

        return DatasetSearchResponse(
            datasets=datasets,
            total=len(datasets),
            query_time_ms=query_time_ms
        )

    except ExternalDependencyError as e:
        audit_logger.log_data_access(
            user_id=current_user.user_id,
            dataset_id=request.query,
            action="search_failed",
            request=http_request,
            details={"query": request.query, "error": str(e)}
        )
        raise HTTPException(
            status_code=503,
            detail=f"Data service unavailable: {str(e)}"
        )
    except Exception as e:
        audit_logger.log_data_access(
            user_id=current_user.user_id,
            dataset_id=request.query,
            action="search_failed",
            request=http_request,
            details={"query": request.query, "error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
