"""
DeepAnalyze Client - Integration with YOLO Auto-Training
Location: business-api/src/api/deepanalyze_client.py

This module provides integration with DeepAnalyze API for data analysis.
"""

import os
import requests
from typing import Optional, List, Dict, Any
from pathlib import Path


class DeepAnalyzeClient:
    """
    Client for DeepAnalyze API.

    Supports both local deployment and hosted API.
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = "DeepAnalyze-8B"
    ):
        """
        Initialize DeepAnalyze client.

        Args:
            base_url: API base URL (e.g., http://localhost:8200/v1)
            api_key: API key for hosted service
            model: Model name to use
        """
        self.base_url = base_url or os.getenv(
            "DEEPANALYZE_API_URL",
            "http://localhost:8200/v1"
        )
        self.api_key = api_key or os.getenv("DEEPANALYZE_API_KEY", "dummy")
        self.model = model
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def health_check(self) -> bool:
        """Check if DeepAnalyze API is available."""
        try:
            response = requests.get(f"{self.base_url.replace('/v1', '')}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def upload_file(self, file_path: str) -> Optional[str]:
        """
        Upload a file for analysis.

        Args:
            file_path: Path to file to upload

        Returns:
            File ID if successful, None otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                response = self._session.post(
                    f"{self.base_url}/files",
                    files={'file': (Path(file_path).name, f)}
                )

            if response.status_code == 200:
                return response.json().get('id')
            return None
        except Exception as e:
            print(f"Error uploading file: {e}")
            return None

    def analyze(
        self,
        prompt: str,
        file_ids: List[str] = None,
        thread_id: str = None,
        temperature: float = 0.4,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send analysis request to DeepAnalyze.

        Args:
            prompt: Analysis prompt/instruction
            file_ids: List of file IDs to analyze
            thread_id: Thread ID for multi-turn conversation
            temperature: Sampling temperature
            stream: Enable streaming response

        Returns:
            Response dictionary with content and generated files
        """
        messages = [{"role": "user", "content": prompt}]

        # Add file IDs to message if provided
        if file_ids:
            messages[0]["file_ids"] = file_ids

        # Add thread ID to latest message if provided
        if thread_id:
            messages[-1]["thread_id"] = thread_id

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }

        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                stream=stream
            )

            if stream:
                return {"stream": response.iter_lines()}
            else:
                result = response.json()
                return {
                    "content": result["choices"][0]["message"]["content"],
                    "thread_id": result["choices"][0]["message"].get("thread_id"),
                    "files": result.get("generated_files", [])
                }
        except Exception as e:
            return {"error": str(e)}

    def analyze_dataset(
        self,
        dataset_path: str,
        analysis_type: str = "quality"
    ) -> Dict[str, Any]:
        """
        Analyze a dataset with predefined analysis prompts.

        Args:
            dataset_path: Path to dataset file or directory
            analysis_type: Type of analysis (quality, distribution, anomalies)

        Returns:
            Analysis results
        """
        # Define analysis prompts based on type
        prompts = {
            "quality": "Analyze this dataset for quality issues: check for missing values, outliers, duplicates, and data type inconsistencies. Provide a detailed quality report.",
            "distribution": "Analyze the data distribution: statistics, histograms, correlations between features, and identify any patterns or trends.",
            "anomalies": "Perform anomaly detection: identify unusual patterns, outliers, and potential data errors.",
            "full": "Perform a comprehensive data analysis including: data quality, distribution, correlations, anomalies, and provide actionable insights."
        }

        prompt = prompts.get(analysis_type, prompts["quality"])

        # For directory, analyze all relevant files
        path = Path(dataset_path)
        if path.is_dir():
            files = list(path.glob("*.csv")) + list(path.glob("*.xlsx")) + list(path.glob("*.json"))
        else:
            files = [path]

        # Upload files and get IDs
        file_ids = []
        for f in files:
            file_id = self.upload_file(str(f))
            if file_id:
                file_ids.append(file_id)

        if not file_ids:
            return {"error": "No files uploaded successfully"}

        # Send analysis request
        return self.analyze(prompt, file_ids=file_ids)

    def generate_report(
        self,
        data_description: str,
        analysis_goals: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive data science report.

        Args:
            data_description: Description of the data
            analysis_goals: List of analysis objectives

        Returns:
            Generated report
        """
        prompt = f"""# Data Science Report Generation

## Data Description
{data_description}

## Analysis Objectives
{chr(10).join(f"- {goal}" for goal in analysis_goals)}

Please perform comprehensive analysis and generate a detailed report with visualizations."""

        return self.analyze(prompt)
