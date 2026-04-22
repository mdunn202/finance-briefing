from __future__ import annotations

import logging

from anthropic import AsyncAnthropic

from agents.filter import ScoredArticle
from agents.ingestion import UserProfile
from config import get_settings

logger = logging.getLogger(__name__)

# Minimum relevance score for an article to be included in the briefing
RELEVANCE_THRESHOLD = 5


class SynthesisAgent:

    def __init__(self) -> None:
        self.client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)

    def _build_system_prompt(self, profile: UserProfile) -> str:
        return f"""You are a professional news briefing writer specialising in the {profile.vertical} vertical.

Your job is to write a concise, engaging audio news briefing for {profile.name} based on today's top stories.

Rules:
- Write in natural spoken English — this will be read aloud by a text-to-speech engine
- No bullet points, no markdown, no headers, no lists
- No phrases like "In this briefing" or "Let's get started" — just dive straight into the news
- Group related stories naturally where possible
- Keep the total briefing to 3-5 minutes of speaking time (approximately 450-750 words)
- End with a single sentence that gives a forward-looking thought or theme to watch
- Be accurate — only state what the articles say, do not embellish or invent details"""

    def _build_user_message(self, articles: list[ScoredArticle]) -> str:
        # Filter by relevance threshold and cap at top 10 to manage context
        filtered = [a for a in articles if a.relevance_score >= RELEVANCE_THRESHOLD][:10]

        if not filtered:
            return "No relevant articles were found for today's briefing."

        lines = ["Write a news briefing based on the following articles, ranked by relevance:\n"]
        for i, article in enumerate(filtered, 1):
            lines.append(
                f"{i}. [{article.relevance_score}/10] {article.title}\n"
                f"   Source: {article.source}\n"
                f"   Summary: {article.description or 'N/A'}\n"
                f"   Relevance: {article.relevance_reason}\n"
            )
        return "\n".join(lines)

    async def run(self, articles: list[ScoredArticle], profile: UserProfile) -> str:
        """Generate a spoken briefing from scored articles."""
        if not articles:
            logger.warning("Synthesis Agent received empty article list")
            return "No articles were available to generate a briefing today."

        logger.info(f"Synthesis Agent generating briefing for {profile.name}")

        user_message = self._build_user_message(articles)

        response = await self.client.messages.create(
            model=get_settings().synthesis_model,
            max_tokens=1024,
            system=self._build_system_prompt(profile),
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        briefing = response.content[0].text.strip()
        logger.info(f"Synthesis Agent generated briefing ({len(briefing.split())} words)")
        return briefing