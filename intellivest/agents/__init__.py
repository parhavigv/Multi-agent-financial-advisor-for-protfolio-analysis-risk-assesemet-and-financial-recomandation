"""Specialist analyst agents."""

from intellivest.agents.base import AgentContext, AnalystAgent, directional_signal
from intellivest.agents.fundamental import FundamentalAnalysisAgent
from intellivest.agents.market import MarketAnalysisAgent
from intellivest.agents.portfolio import PortfolioAnalysisAgent
from intellivest.agents.recommendation import RecommendationAgent
from intellivest.agents.risk import RiskAnalysisAgent
from intellivest.agents.sentiment import SentimentAnalysisAgent


def default_agents() -> list[AnalystAgent]:
    """The specialist roster used by the advisory pipeline."""
    return [
        MarketAnalysisAgent(),
        FundamentalAnalysisAgent(),
        SentimentAnalysisAgent(),
        PortfolioAnalysisAgent(),
        RiskAnalysisAgent(),
    ]


__all__ = [
    "AgentContext",
    "AnalystAgent",
    "FundamentalAnalysisAgent",
    "MarketAnalysisAgent",
    "PortfolioAnalysisAgent",
    "RecommendationAgent",
    "RiskAnalysisAgent",
    "SentimentAnalysisAgent",
    "default_agents",
    "directional_signal",
]
