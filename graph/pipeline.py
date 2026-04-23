from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.ingestion import IngestionAgent, UserProfile, Article
from agents.filter import FilterAgent, ScoredArticle
from agents.synthesis import SynthesisAgent
from agents.tts import TTSAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class BriefingState(TypedDict):
    profile: UserProfile
    raw_articles: list[Article]
    scored_articles: list[ScoredArticle]
    briefing_text: str
    audio_path: Path | None
    error: str | None


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

async def ingest(state: BriefingState) -> BriefingState:
    try:
        agent = IngestionAgent()
        articles = await agent.run(state["profile"])
        return {**state, "raw_articles": articles}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return {**state, "error": str(e)}


async def filter_articles(state: BriefingState) -> BriefingState:
    if state.get("error"):
        return state
    try:
        agent = FilterAgent()
        scored = await agent.run(state["raw_articles"], state["profile"])
        return {**state, "scored_articles": scored}
    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        return {**state, "error": str(e)}


async def synthesise(state: BriefingState) -> BriefingState:
    if state.get("error"):
        return state
    try:
        agent = SynthesisAgent()
        text = await agent.run(state["scored_articles"], state["profile"])
        return {**state, "briefing_text": text}
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return {**state, "error": str(e)}


async def text_to_speech(state: BriefingState) -> BriefingState:
    if state.get("error"):
        return state
    try:
        agent = TTSAgent()
        path = await agent.run(state["briefing_text"], state["profile"].name)
        return {**state, "audio_path": path}
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return {**state, "error": str(e)}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_pipeline() -> StateGraph:
    graph = StateGraph(BriefingState)

    graph.add_node("ingest", ingest)
    graph.add_node("filter_articles", filter_articles)
    graph.add_node("synthesise", synthesise)
    graph.add_node("text_to_speech", text_to_speech)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "filter_articles")
    graph.add_edge("filter_articles", "synthesise")
    graph.add_edge("synthesise", "text_to_speech")
    graph.add_edge("text_to_speech", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point for testing
# ---------------------------------------------------------------------------

async def run_pipeline(profile: UserProfile) -> BriefingState:
    pipeline = build_pipeline()
    initial_state: BriefingState = {
        "profile": profile,
        "raw_articles": [],
        "scored_articles": [],
        "briefing_text": "",
        "audio_path": None,
        "error": None,
    }
    result = await pipeline.ainvoke(initial_state)
    return result