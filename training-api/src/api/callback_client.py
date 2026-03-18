"""
Callback Client - Send task completion notifications to Business API

This module provides a client for notifying the Business API when tasks complete.
"""

from typing import Optional, Dict, Any
import httpx
from pydantic import BaseModel
from datetime import datetime


class CallbackClient:
    """Client for sending callbacks to Business API."""

    def __init__(
        self,
        business_api_url: str,
        api_key: str,
        timeout: int = 30
    ):
        """
        Initialize the callback client.

        Args:
            business_api_url: Base URL of the business API
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.business_api_url = business_api_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def notify_task_complete(
        self,
        task_id: str,
        status: str,
        metrics: Optional[Dict[str, float]] = None,
        model_path: Optional[str] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Notify Business API that a task has completed.

        Args:
            task_id: Task identifier
            status: Task status (completed, failed)
            metrics: Training metrics if completed
            model_path: Path to the trained model
            error: Error message if failed

        Returns:
            Callback response
        """
        payload = {
            "task_id": task_id,
            "status": status,
            "completed_at": datetime.now().isoformat()
        }

        if metrics:
            payload["metrics"] = metrics
        if model_path:
            payload["model_path"] = model_path
        if error:
            payload["error"] = error

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.business_api_url}/api/v1/callback/task/callback",
                    json=payload,
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Failed to send callback for task {task_id}: {e}")
            return {"error": str(e)}


class TaskCallback(BaseModel):
    """Task callback model for sending to Business API."""
    task_id: str
    status: str  # completed, failed
    metrics: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None
    error: Optional[str] = None
    completed_at: Optional[str] = None
