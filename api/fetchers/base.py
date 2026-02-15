"""
Base fetcher class for API data retrieval.
Provides common functionality for all data fetchers.
"""

import asyncio
import aiohttp
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    """Abstract base class for API data fetchers."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the base fetcher.
        
        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self.session
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            APIError: If the request fails
        """
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise APIError(
                        f"API request failed with status {response.status}: {error_text}"
                    )
                return await response.json()
        except aiohttp.ClientError as e:
            raise APIError(f"Network error occurred: {str(e)}")
    
    async def fetch_recent(
        self,
        hours: int = 24,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent data for the specified number of hours.
        
        Args:
            hours: Number of hours to fetch
            end_time: End time (defaults to now)
            
        Returns:
            List of data records
        """
        if end_time is None:
            end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        return await self.fetch_range(start_time, end_time)
    
    @abstractmethod
    async def fetch_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Fetch data for a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of data records
        """
        pass
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class APIError(Exception):
    """Custom exception for API errors."""
    
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class RateLimitError(APIError):
    """Exception raised when API rate limit is exceeded."""
    
    def __init__(self, message: str = "API rate limit exceeded"):
        super().__init__(message, status_code=429)
