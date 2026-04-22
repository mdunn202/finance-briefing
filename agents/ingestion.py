from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel

from config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vertical defaults
# ---------------------------------------------------------------------------

VERTICAL_DEFAULTS: dict[str, dict[str, Any]] = {
    "finance": {
        "categories": ["business", "technology"],
        "keywords": ["interest rates", "inflation", "markets", "FTSE", "GDP", "Fed", "Bank of England"],
    },
    "healthcare": {
        "categories": ["health", "science"],
        "keywords": ["NHS", "clinical trials", "drug approval", "health policy"],
    },
    "technology": {
        "categories": ["technology", "science"],
        "keywords": ["AI", "cybersecurity", "regulation", "startups"],
    },
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    name: str
    vertical: str
    categories: list[str] = []
    keywords: list[str] = []
    premium_rss_feeds: list[str] = []  # stubbed for future use

    def model_post_init(self, __context: Any) -> None:
        defaults = VERTICAL_DEFAULTS.get(self.vertical, {})
        if not self.categories:
            self.categories = defaults.get("categories", [])
        if not self.keywords:
            self.keywords = defaults.get("keywords", [])


class Article(BaseModel):
    title: str
    description: str | None
    url: str
    source: str
    published_at: str
    content: str | None


# ---------------------------------------------------------------------------
# Ingestion agent
# ---------------------------------------------------------------------------

class IngestionAgent:
    BASE_URL = "https://newsapi.org/v2"

    def __init__(self) -> None:
        self.api_key = get_settings().news_api_key

    async def _get_sources(self, categories: list[str]) -> list[str]:
        """Fetch vetted source IDs from NewsAPI for the given categories."""
        source_ids: list[str] = []

        async with httpx.AsyncClient() as client:
            for category in categories:
                response = await client.get(
                    f"{self.BASE_URL}/sources",
                    params={
                        "apiKey": self.api_key,
                        "category": category,
                        "language": "en",
                    },
                )
                response.raise_for_status()
                data = response.json()
                ids = [s["id"] for s in data.get("sources", [])]
                source_ids.extend(ids)
                logger.info(f"Category '{category}': found {len(ids)} sources")

        return list(set(source_ids))  # deduplicate

    async def _get_headlines(self, source_ids: list[str], page_size: int = 20) -> list[Article]:
        """Fetch top headlines for the given source IDs."""
        # NewsAPI accepts max 20 source IDs per request
        chunks = [source_ids[i:i + 20] for i in range(0, len(source_ids), 20)]
        articles: list[Article] = []

        async with httpx.AsyncClient() as client:
            for chunk in chunks:
                response = await client.get(
                    f"{self.BASE_URL}/top-headlines",
                    params={
                        "apiKey": self.api_key,
                        "sources": ",".join(chunk),
                        "pageSize": page_size,
                    },
                )
                response.raise_for_status()
                data = response.json()

                for item in data.get("articles", []):
                    articles.append(Article(
                        title=item.get("title") or "",
                        description=item.get("description"),
                        url=item.get("url") or "",
                        source=item.get("source", {}).get("name") or "",
                        published_at=item.get("publishedAt") or "",
                        content=item.get("content"),
                    ))

        logger.info(f"Fetched {len(articles)} articles total")
        return articles

    async def run(self, profile: UserProfile) -> list[Article]:
        """Main entry point — returns raw articles for the Filter Agent."""
        logger.info(f"Running ingestion for profile: {profile.name} | vertical: {profile.vertical}")
        source_ids = await self._get_sources(profile.categories)

        if not source_ids:
            logger.warning("No sources found for the given categories")
            return []

        articles = await self._get_headlines(source_ids)
        return articles