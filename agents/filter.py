from __future__ import annotations

import logging
import json
from typing import Any

from anthropic import AsyncAnthropic

from agents.ingestion import Article, UserProfile
from config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ScoredArticle(Article):
    relevance_score: int        # 1-10
    relevance_reason: str       # brief explanation from Claude


# ---------------------------------------------------------------------------
# Filter Agent
# ---------------------------------------------------------------------------

class FilterAgent:

    def __init__(self) -> None:
        self.client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)

    def _build_system_prompt(self, profile: UserProfile) -> str:
        return f"""You are a news relevance scoring agent.

Your job is to score news articles for relevance to a user's professional vertical and interests.

User profile:
- Name: {profile.name}
- Vertical: {profile.vertical}
- Keywords of interest: {", ".join(profile.keywords)}

Scoring criteria:
- 8-10: Directly relevant. Covers the user's vertical, key themes, or named keywords. Would clearly belong in a professional briefing.
- 5-7: Tangentially relevant. Related industry, macro context, or adjacent topic that a professional in this vertical would care about.
- 1-4: Not relevant. General news with no meaningful connection to the user's vertical or interests.

You must respond with a JSON array and nothing else. No preamble, no markdown, no explanation outside the array.

Format:
[
  {{
    "url": "article url",
    "relevance_score": 8,
    "relevance_reason": "one sentence explanation"
  }}
]"""

    def _build_user_message(self, articles: list[Article]) -> str:
        lines = ["Score the following articles:\n"]
        for i, article in enumerate(articles, 1):
            lines.append(
                f"{i}. URL: {article.url}\n"
                f"   Title: {article.title}\n"
                f"   Description: {article.description or 'N/A'}\n"
            )
        return "\n".join(lines)

    async def run(self, articles: list[Article], profile: UserProfile) -> list[ScoredArticle]:
        """Score all articles and return them with relevance scores attached."""
        if not articles:
            logger.warning("Filter Agent received empty article list")
            return []

        logger.info(f"Filter Agent scoring {len(articles)} articles for vertical: {profile.vertical}")

        response = await self.client.messages.create(
            model=get_settings().filter_model,
            max_tokens=2048,
            system=self._build_system_prompt(profile),
            messages=[
                {"role": "user", "content": self._build_user_message(articles)}
            ]
        )

        raw = response.content[0].text.strip()

        try:
            scores: list[dict[str, Any]] = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Filter Agent failed to parse Claude response: {e}\nRaw: {raw}")
            return []

        # Build a lookup by URL for fast matching
        article_map = {a.url: a for a in articles}
        scored: list[ScoredArticle] = []

        for item in scores:
            url = item.get("url")
            article = article_map.get(url)
            if not article:
                logger.warning(f"Filter Agent returned unknown URL: {url}")
                continue
            scored.append(ScoredArticle(
                **article.model_dump(),
                relevance_score=item.get("relevance_score", 1),
                relevance_reason=item.get("relevance_reason", ""),
            ))

        # Sort highest score first
        scored.sort(key=lambda a: a.relevance_score, reverse=True)
        logger.info(f"Filter Agent returned {len(scored)} scored articles")
        return scored