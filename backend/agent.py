"""Agentic Orchestration (Search-Scoped CRAG/Self-RAG).

Defines the LangGraph state machine orchestrating the Search-Scoped Hybrid RAG
pipeline. Uses Tavily web search (scoped to en.wikipedia.org) to identify
relevant Wikipedia articles, then performs article-scoped hybrid retrieval in
Qdrant. Implements batched document grading, hallucination checking, and answer
quality loops with separate retry counters and a hard step budget.
"""

import json
import logging
from typing import Dict, List, Literal, TypedDict
from urllib.parse import unquote, urlparse

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from tavily import TavilyClient

from backend.config import get_settings
from backend.llmops import get_langfuse_handler, safe_generate
from backend.models import QueryStrategies
from backend.query_expansion import expand_query
from backend.retrieval import extract_title_from_wikipedia_url, hybrid_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """The state dictionary for the LangGraph agent."""
    query: str
    expanded_queries: List[str]
    target_articles: List[str]
    documents: List[Dict]
    web_snippets: List[Dict]
    generation: str
    retrieval_grade: str
    hallucination_grade: str
    answer_grade: str
    steps: int
    active_strategies: QueryStrategies
    hallucination_retries: int
    answer_retries: int


# ---------------------------------------------------------------------------
# Helper: Get LLM Instance
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.0, max_tokens: int = 256) -> ChatOpenAI:
    """Return a ChatOpenAI instance configured for OpenRouter."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Node Definitions
# ---------------------------------------------------------------------------

async def node_expand_query(state: AgentState) -> Dict:
    """Node: Expand the original query using active strategies."""
    query = state["query"]
    strategies = state["active_strategies"]
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: EXPAND QUERY (step %d) ---", steps)
    expanded_queries = await expand_query(query, strategies)

    return {"expanded_queries": expanded_queries, "steps": steps}


async def node_identify_articles(state: AgentState) -> Dict:
    """Node: Use Tavily web search scoped to en.wikipedia.org to identify
    the most relevant Wikipedia article(s) for the query.

    Extracts article titles from Wikipedia URLs in the search results.
    Also stores the web snippets as fallback context in case the scoped
    Qdrant search returns no results (article not yet ingested).
    """
    query = state["query"]
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: IDENTIFY ARTICLES (step %d) ---", steps)

    settings = get_settings()
    if not settings.tavily_api_key:
        logger.warning("Tavily API key not configured. Skipping article identification.")
        return {"target_articles": [], "web_snippets": [], "steps": steps}

    client = TavilyClient(api_key=settings.tavily_api_key)

    try:
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=settings.tavily_max_results,
            include_domains=["en.wikipedia.org"],
        )

        target_articles = []
        web_snippets = []

        for result in response.get("results", []):
            url = result.get("url", "")
            title = extract_title_from_wikipedia_url(url)
            if title and title not in target_articles:
                target_articles.append(title)

            # Store the snippet as fallback context
            web_snippets.append({
                "id": url,
                "title": title or result.get("title", "Web Result"),
                "content": result.get("content", ""),
                "url": url,
                "score": result.get("score", 0.0),
            })

        logger.info(
            "Tavily identified %d Wikipedia article(s): %s",
            len(target_articles),
            ", ".join(target_articles),
        )

        return {
            "target_articles": target_articles,
            "web_snippets": web_snippets,
            "steps": steps,
        }

    except Exception as exc:
        logger.error("Tavily article identification failed: %s", exc)
        return {"target_articles": [], "web_snippets": [], "steps": steps}


async def node_retrieve(state: AgentState) -> Dict:
    """Node: Retrieve documents using article-scoped hybrid search.

    Uses the article titles identified by Tavily to filter the Qdrant
    search. If no articles were identified, falls back to unscoped search.
    """
    queries_to_search = state.get("expanded_queries", [state["query"]])
    target_articles = state.get("target_articles", [])
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: RETRIEVE (step %d) ---", steps)

    settings = get_settings()
    all_documents = []
    seen_ids = set()

    for q in queries_to_search:
        docs, _ = await hybrid_search(
            q,
            article_titles=target_articles if target_articles else None,
        )
        for doc in docs:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                all_documents.append(doc)

    # Sort by reranker score descending
    all_documents.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # Cap at max_generation_docs to keep the context focused
    final_docs = all_documents[:settings.max_generation_docs]

    logger.info(
        "Retrieve produced %d unique chunks (capped to %d).",
        len(all_documents),
        len(final_docs),
    )

    # If scoped search returned nothing but we have web snippets, use those
    if not final_docs and state.get("web_snippets"):
        logger.info("Scoped search returned no results. Using Tavily web snippets as fallback.")
        final_docs = state["web_snippets"][:settings.max_generation_docs]

    return {"documents": final_docs, "steps": steps}


async def node_grade_documents(state: AgentState) -> Dict:
    """Node: Evaluate document relevance using a single batched LLM call.

    Instead of making N serial LLM calls (one per document), concatenates
    all documents with numbered indices and asks the LLM to return a
    comma-separated list of relevant document numbers.
    """
    query = state["query"]
    documents = state.get("documents", [])
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: GRADE DOCUMENTS (step %d) ---", steps)

    if not documents:
        return {"documents": [], "retrieval_grade": "irrelevant", "steps": steps}

    llm = _get_llm(temperature=0.0, max_tokens=100)

    # Build a numbered list of document snippets for batched grading
    doc_summaries = []
    for i, doc in enumerate(documents):
        content = doc.get("content", "")[:300]
        doc_summaries.append(f"[{i}] Title: {doc.get('title', 'Unknown')}\n{content}")

    docs_text = "\n\n".join(doc_summaries)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a relevance grader. Given a user question and a numbered list of "
         "retrieved document snippets, identify which documents are relevant to "
         "answering the question.\n"
         "Return ONLY a comma-separated list of the relevant document numbers "
         "(e.g., '0,2,4'). If none are relevant, return 'NONE'. "
         "Do not include any other text."),
        ("user",
         "User question: {query}\n\n"
         "Documents:\n{documents}\n\n"
         "Relevant document numbers:"),
    ])

    chain = prompt | llm

    try:
        res = await chain.ainvoke({"query": query, "documents": docs_text})
        content = (res.content if hasattr(res, "content") else str(res)).strip()

        if "none" in content.lower():
            logger.info("Batched grading: no documents deemed relevant.")
            return {"documents": [], "retrieval_grade": "irrelevant", "steps": steps}

        # Parse the comma-separated indices
        relevant_indices = set()
        for part in content.replace(" ", "").split(","):
            try:
                idx = int(part)
                if 0 <= idx < len(documents):
                    relevant_indices.add(idx)
            except ValueError:
                continue

        filtered_docs = [documents[i] for i in sorted(relevant_indices)]

        grade = "relevant" if filtered_docs else "irrelevant"
        logger.info(
            "Batched grading result: %s (%d kept out of %d)",
            grade, len(filtered_docs), len(documents),
        )

        return {"documents": filtered_docs, "retrieval_grade": grade, "steps": steps}

    except Exception as exc:
        logger.warning("Batched grading failed: %s. Keeping all documents.", exc)
        return {"documents": documents, "retrieval_grade": "relevant", "steps": steps}


async def node_generate_from_web(state: AgentState) -> Dict:
    """Node: Generate a response using Tavily web snippets as fallback context.

    Called when scoped retrieval returns no relevant documents and the
    pipeline falls back to the web search snippets collected during
    article identification.
    """
    query = state["query"]
    web_snippets = state.get("web_snippets", [])
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: GENERATE FROM WEB SNIPPETS (step %d) ---", steps)

    if web_snippets:
        context = "\n\n".join(
            f"Title: {d.get('title')}\nContent: {d.get('content')}"
            for d in web_snippets
        )
    else:
        context = "No relevant context was found."

    generation = await safe_generate(query=query, context=context)
    return {"generation": generation, "documents": web_snippets, "steps": steps}


async def node_generate(state: AgentState) -> Dict:
    """Node: Generate response using NeMo Guardrails with retrieved context."""
    query = state["query"]
    documents = state.get("documents", [])
    steps = state.get("steps", 0) + 1

    logger.info("--- NODE: GENERATE (step %d) ---", steps)

    context = "\n\n".join(
        f"Title: {d.get('title')}\nContent: {d.get('content')}"
        for d in documents
    )

    generation = await safe_generate(query=query, context=context)
    return {"generation": generation, "steps": steps}


async def node_check_hallucination(state: AgentState) -> Dict:
    """Node: Evaluate if the generation is grounded in the retrieved documents."""
    documents = state.get("documents", [])
    generation = state["generation"]
    steps = state.get("steps", 0) + 1
    retries = state.get("hallucination_retries", 0)

    logger.info("--- NODE: CHECK HALLUCINATION (step %d) ---", steps)

    llm = _get_llm(temperature=0.0, max_tokens=10)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grader assessing whether an LLM generation is grounded in "
         "a set of retrieved facts. Give a binary score 'yes' or 'no'. "
         "'Yes' means that the answer is grounded in the facts."),
        ("user",
         "Set of facts:\n\n{documents}\n\n"
         "LLM generation: {generation}\n\n"
         "Score (yes/no):"),
    ])

    context = "\n\n".join(
        f"Title: {d.get('title')}\nContent: {d.get('content')}"
        for d in documents
    )
    chain = prompt | llm

    try:
        res = await chain.ainvoke({"documents": context, "generation": generation})
        grade = (res.content if hasattr(res, "content") else str(res)).strip().lower()

        if "yes" in grade:
            logger.info("Hallucination check passed (grounded).")
            return {"hallucination_grade": "grounded", "steps": steps}
        else:
            logger.warning("Hallucination check failed (not grounded). Retry %d.", retries + 1)
            return {
                "hallucination_grade": "hallucinated",
                "steps": steps,
                "hallucination_retries": retries + 1,
            }
    except Exception as exc:
        logger.warning("Hallucination check failed with error: %s. Passing through.", exc)
        return {"hallucination_grade": "grounded", "steps": steps}


async def node_check_answer_quality(state: AgentState) -> Dict:
    """Node: Evaluate if the generation answers the original query."""
    query = state["query"]
    generation = state["generation"]
    steps = state.get("steps", 0) + 1
    retries = state.get("answer_retries", 0)

    logger.info("--- NODE: CHECK ANSWER QUALITY (step %d) ---", steps)

    llm = _get_llm(temperature=0.0, max_tokens=10)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grader assessing whether an answer addresses a question. "
         "Give a binary score 'yes' or 'no'. 'Yes' means the answer resolves the question."),
        ("user",
         "User question:\n\n{query}\n\n"
         "LLM generation: {generation}\n\n"
         "Score (yes/no):"),
    ])

    chain = prompt | llm

    try:
        res = await chain.ainvoke({"query": query, "generation": generation})
        grade = (res.content if hasattr(res, "content") else str(res)).strip().lower()

        if "yes" in grade:
            logger.info("Answer quality check passed (useful).")
            return {"answer_grade": "useful", "steps": steps}
        else:
            logger.warning("Answer quality check failed (not useful). Retry %d.", retries + 1)
            return {
                "answer_grade": "not_useful",
                "steps": steps,
                "answer_retries": retries + 1,
            }
    except Exception as exc:
        logger.warning("Answer quality check failed with error: %s. Passing through.", exc)
        return {"answer_grade": "useful", "steps": steps}


# ---------------------------------------------------------------------------
# Conditional Edges (with step budget enforcement)
# ---------------------------------------------------------------------------

def _is_over_budget(state: AgentState) -> bool:
    """Check if the graph has exceeded its step budget."""
    settings = get_settings()
    return state.get("steps", 0) >= settings.max_graph_steps


def route_after_grading(state: AgentState) -> Literal["generate_from_web", "generate"]:
    """Route based on document relevance after grading.

    If no relevant documents remain, fall back to generating from
    the Tavily web snippets collected during article identification.
    """
    if state.get("retrieval_grade") == "irrelevant":
        return "generate_from_web"
    return "generate"


def route_after_hallucination(
    state: AgentState,
) -> Literal["generate", "check_answer_quality"]:
    """Route based on grounding. Retry generation if hallucinated (max 2 retries).

    Does not loop back to retrieve -- the documents are already scoped
    and reranked. Instead, retries the generation step with the same context.
    """
    if _is_over_budget(state):
        logger.warning("Step budget exhausted. Forcing answer quality check.")
        return "check_answer_quality"

    if (
        state.get("hallucination_grade") == "hallucinated"
        and state.get("hallucination_retries", 0) < 2
    ):
        return "generate"
    return "check_answer_quality"


def route_after_answer_quality(state: AgentState) -> Literal["expand_query", "__end__"]:
    """Route based on answer usefulness. Expand query if not useful (max 2 retries)."""
    if _is_over_budget(state):
        logger.warning("Step budget exhausted. Terminating graph.")
        return END

    if (
        state.get("answer_grade") == "not_useful"
        and state.get("answer_retries", 0) < 2
    ):
        return "expand_query"
    return END


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------

def compile_agent_graph():
    """Compile the LangGraph state machine workflow.

    Graph topology:
        expand_query -> identify_articles -> retrieve -> grade_documents
            -> [if irrelevant] -> generate_from_web -> END
            -> [if relevant]   -> generate -> check_hallucination
                -> [if hallucinated, retries < 2] -> generate (retry)
                -> [if grounded] -> check_answer_quality
                    -> [if not useful, retries < 2] -> expand_query (re-expand and re-retrieve)
                    -> [if useful or budget exhausted] -> END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("expand_query", node_expand_query)
    workflow.add_node("identify_articles", node_identify_articles)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("grade_documents", node_grade_documents)
    workflow.add_node("generate_from_web", node_generate_from_web)
    workflow.add_node("generate", node_generate)
    workflow.add_node("check_hallucination", node_check_hallucination)
    workflow.add_node("check_answer_quality", node_check_answer_quality)

    # Set entry point
    workflow.set_entry_point("expand_query")

    # Standard edges
    workflow.add_edge("expand_query", "identify_articles")
    workflow.add_edge("identify_articles", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("generate_from_web", END)
    workflow.add_edge("generate", "check_hallucination")

    # Conditional edges
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate_from_web": "generate_from_web",
            "generate": "generate",
        },
    )

    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {
            "generate": "generate",
            "check_answer_quality": "check_answer_quality",
        },
    )

    workflow.add_conditional_edges(
        "check_answer_quality",
        route_after_answer_quality,
        {
            "expand_query": "expand_query",
            END: END,
        },
    )

    app = workflow.compile()
    logger.info("Search-Scoped CRAG/Self-RAG workflow compiled successfully.")
    return app


# Singleton graph instance
agent_app = compile_agent_graph()
