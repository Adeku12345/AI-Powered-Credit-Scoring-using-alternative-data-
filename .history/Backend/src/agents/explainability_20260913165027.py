from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from backend.agents.state import CreditScoringState
from backend.core.llm import llm
from backend.core.logging import setup_logger

logger = setup_logger("agent-explainer")

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a Credit Analyst. Explain clearly: overall rating, strengths, weaknesses, 1-2 actionable steps. Professional, warm, concise. Avoid jargon."),
    ("human", "Score: {score}/850, Grade: {grade}, Risk: {risk}. Factors: {factors}. Alerts: {alerts}.")
])

def _fallback(state):
    s, g, r = state["final_score"], state["grade"], state["risk_category"]
    rating = "Excellent" if s>=780 else "Good" if s>=720 else "Fair" if s>=660 else "Needs Improvement"
    return f"Your credit score of {s} ({g}) is considered {rating}. Risk: {r}. Continue paying on time and keep debt levels low to improve further."

async def explainer_node(state: CreditScoringState) -> Dict:
    s, g, r = state["final_score"], state["grade"], state["risk_category"]
    alerts = ", ".join(state.get("fraud_flags",[])) or "None"
    all_factors = []
    for ag in state["agent_scores"].values():
        for f in ag.get("factors",[]):
            if isinstance(f.get("impact"), (int,float)) and abs(f["impact"])>0.5: all_factors.append(f)
    all_factors.sort(key=lambda x:-abs(x["impact"]))
    top5 = all_factors[:5]
    factors_text = "; ".join(f"{f['factor']}: {f.get('value','')} ({'+' if f['impact']>0 else ''}{f['impact']:.1f}pts)" for f in top5) or "Standard profile"

    summary = None
    if llm:
        try:
            chain = PROMPT | llm
            resp = await chain.ainvoke({"score":s,"grade":g,"risk":r,"factors":factors_text,"alerts":alerts})
            summary = resp.content.strip()
        except Exception as e:
            logger.warning(f"LLM failed: {e}")

    return {"explanation": {"summary": summary or _fallback(state), "llm_generated": bool(summary), "top_factors": top5}}

explainer_runnable = RunnableLambda(explainer_node)