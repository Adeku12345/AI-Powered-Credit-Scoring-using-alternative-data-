from langgraph.graph import StateGraph, END
from backend.agents.state import CreditScoringState
from backend.agents.payment_agent import payment_runnable
from backend.agents.finance_agent import finance_runnable
from backend.agents.history_agent import history_runnable
from backend.agents.alternative_agent import alternative_runnable
from backend.agents.fraud_agent import fraud_runnable
from backend.agents.coordinator import coordinator_runnable
from backend.agents.explainer_agent import explainer_runnable
from backend.agents.fairness_agent import fairness_runnable
from backend.core.logging import setup_logger

logger = setup_logger("langgraph")

workflow = StateGraph(CreditScoringState)
workflow.add_node("payment", payment_runnable)
workflow.add_node("finance", finance_runnable)
workflow.add_node("history", history_runnable)
workflow.add_node("alternative", alternative_runnable)
workflow.add_node("fraud", fraud_runnable)
workflow.add_node("coordinator", coordinator_runnable)
workflow.add_node("explainer", explainer_runnable)
workflow.add_node("fairness", fairness_runnable)

workflow.set_entry_point("payment")
workflow.add_edge("payment", "finance")
workflow.add_edge("payment", "history")
workflow.add_edge("payment", "alternative")
workflow.add_edge("payment", "fraud")
workflow.add_edge(["finance","history","alternative","fraud"], "coordinator")
workflow.add_edge("coordinator", "explainer")
workflow.add_edge("explainer", "fairness")
workflow.add_edge("fairness", END)

graph = workflow.compile()
logger.info("✅ LangGraph Multi-Agent Workflow Compiled")