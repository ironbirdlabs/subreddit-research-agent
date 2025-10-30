"""
AgentSystems Subreddit Research Agent

This agent analyzes Reddit communities to answer specific research questions.
It uses a 7-stage LangGraph pipeline to:
  1. Generate targeted search keywords
  2. Fetch relevant Reddit threads
  3. Filter threads for relevance
  4. Analyze individual threads and comments
  5. Synthesize findings across all threads
  6. Generate strategic recommendations
  7. Create professional reports (JSON, Markdown, PDF)

Required endpoints (do not remove):
  POST /invoke    - Main agent logic
  GET  /health    - Container health check
  GET  /metadata  - Agent information
"""

import os
import logging
import pathlib
import json
from datetime import datetime, timedelta, timezone
from typing import TypedDict, List, Dict, Any

import praw
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

from agentsystems_toolkit import get_model
from markdown_pdf import MarkdownPdf, Section


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Setup and Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load and merge agent metadata from agent.yaml + metadata.yaml
agent_identity = yaml.safe_load(
    pathlib.Path(__file__).with_name("agent.yaml").read_text()
)
agent_metadata = yaml.safe_load(
    pathlib.Path(__file__).with_name("metadata.yaml").read_text()
)

# Merge metadata (metadata.yaml takes precedence on conflicts)
meta: Dict[str, Any] = {**agent_identity, **agent_metadata}

app = FastAPI(title=meta.get("name", "Agent"), version=meta.get("version", "0.1.0"))

# Artifacts directory for storing output files
ARTIFACTS_ROOT = pathlib.Path("/artifacts")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reddit API Client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RedditFetcher:
    """Handles Reddit API interactions"""

    def __init__(self):
        """Initialize Reddit API client"""
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'SubredditResearchAgent/0.1.0')
        )

        # Test authentication
        try:
            self.reddit.user.me()
        except Exception:
            # We're in read-only mode (no user context needed for public data)
            pass

    def search_subreddit(
        self,
        subreddit_name: str,
        query: str,
        time_filter: str = 'month',
        limit: int = 10,
        sort: str = 'relevance'
    ) -> List[Dict[str, Any]]:
        """
        Search a subreddit for posts matching a query
        """
        subreddit = self.reddit.subreddit(subreddit_name)

        search_results = subreddit.search(
            query,
            time_filter=time_filter,
            limit=limit,
            sort=sort
        )

        posts = []
        for submission in search_results:
            post_data = {
                'id': submission.id,
                'title': submission.title,
                'author': str(submission.author),
                'score': submission.score,
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc,
                'url': f"https://reddit.com{submission.permalink}",
                'selftext': submission.selftext[:500] if submission.selftext else '[Link post]',
                'is_self': submission.is_self,
                'upvote_ratio': submission.upvote_ratio
            }
            posts.append(post_data)

        return posts

    def fetch_post_comments(
        self,
        post_id: str,
        max_comments: int = 50,
        min_score: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Fetch comments for a specific post
        """
        submission = self.reddit.submission(id=post_id)

        # Replace MoreComments objects to get all comments
        submission.comments.replace_more(limit=0)

        # Flatten comment tree
        all_comments = submission.comments.list()

        # Filter by score
        filtered_comments = [
            c for c in all_comments
            if hasattr(c, 'body') and c.score >= min_score
        ]

        # Sort by score (top comments first)
        sorted_comments = sorted(
            filtered_comments,
            key=lambda x: x.score,
            reverse=True
        )[:max_comments]

        comments = []
        for comment in sorted_comments:
            comment_data = {
                'id': comment.id,
                'author': str(comment.author),
                'body': comment.body,
                'score': comment.score,
                'created_utc': comment.created_utc,
                'num_replies': len(comment.replies) if hasattr(comment, 'replies') else 0,
                'is_submitter': comment.is_submitter
            }
            comments.append(comment_data)

        return comments


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API Models - Define your request/response contract
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class InvokeRequest(BaseModel):
    """Request payload sent to the agent."""
    subreddit: str
    research_question: str
    time_period_days: int = 7


class InvokeResponse(BaseModel):
    """Response returned by the agent."""
    thread_id: str
    subreddit: str
    research_question: str
    report_json: Dict[str, Any]
    report_markdown: str
    report_pdf_base64: str  # Base64-encoded PDF
    timestamp: datetime


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph State - Define what data flows through your agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ResearchState(TypedDict):
    """State for the Reddit Research Agent"""
    # Input parameters
    subreddit: str
    research_question: str
    time_period_days: int

    # Intermediate data
    keywords: List[str]
    all_threads: List[Dict[str, Any]]
    relevant_threads: List[Dict[str, Any]]
    thread_analyses: List[Dict[str, Any]]

    # Output data
    synthesis: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    report_json: Dict[str, Any]
    report_markdown: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

model = get_model("claude-sonnet-4-5", "langchain", temperature=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph Nodes - Implement your business logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_keywords_node(state: ResearchState) -> ResearchState:
    """
    Node 1: Generate targeted search keywords based on the research question
    """
    logger.info("=== STEP 1: Generating Keywords ===")

    prompt_template = """
Generate 7-10 search keywords/phrases that would help find relevant Reddit discussions.

Research Question: {research_question}
Subreddit: r/{subreddit}

IMPORTANT: Be SPECIFIC and use multi-word phrases to reduce irrelevant results.

Include:
- Direct multi-word phrases related to the question (e.g., "AI agent marketplace" not just "AI")
- Problem-focused queries (e.g., "AI agent issues", "marketplace concerns")
- Alternative phrasings of the core concept
- Related technical terms
- Comparison phrases (e.g., "X vs Y")

Return ONLY valid JSON:
{{
  "keywords": [
    "keyword phrase 1",
    "keyword phrase 2",
    ...
  ]
}}
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["research_question", "subreddit"]
    )

    json_parser = JsonOutputParser()
    chain = prompt | model | json_parser

    result = chain.invoke({
        "research_question": state["research_question"],
        "subreddit": state["subreddit"]
    })

    state["keywords"] = result.get("keywords", [])
    logger.info(f"Generated {len(state['keywords'])} keywords")

    return state


def fetch_threads_node(state: ResearchState) -> ResearchState:
    """
    Node 2: Fetch Reddit threads using generated keywords
    """
    logger.info("=== STEP 2: Fetching Threads ===")

    fetcher = RedditFetcher()

    # Determine time filter based on time_period_days
    days = state["time_period_days"]
    if days <= 1:
        time_filter = 'day'
    elif days <= 7:
        time_filter = 'week'
    elif days <= 30:
        time_filter = 'month'
    elif days <= 365:
        time_filter = 'year'
    else:
        time_filter = 'all'

    all_threads = []
    seen_ids = set()

    for keyword in state["keywords"]:
        try:
            threads = fetcher.search_subreddit(
                subreddit_name=state["subreddit"],
                query=keyword,
                time_filter=time_filter,
                limit=10,
                sort='relevance'
            )

            # Deduplicate
            for thread in threads:
                if thread['id'] not in seen_ids:
                    seen_ids.add(thread['id'])
                    all_threads.append(thread)

        except Exception as e:
            logger.error(f"Error searching for '{keyword}': {e}")
            continue

    state["all_threads"] = all_threads
    logger.info(f"Fetched {len(all_threads)} unique threads")

    return state


def filter_threads_node(state: ResearchState) -> ResearchState:
    """
    Node 3: Filter threads for relevance using batch processing
    """
    logger.info("=== STEP 3: Filtering Threads ===")

    all_threads = state["all_threads"]

    if not all_threads:
        logger.warning("No threads to filter")
        state["relevant_threads"] = []
        return state

    prompt_template = """
You are filtering Reddit threads for relevance to a research question.

Research Question: {research_question}

Threads to evaluate:
{threads_batch}

For each thread, determine if it's relevant and assign a relevance score (1-10).
A thread is relevant if it contains discussion that could help answer the research question.

Return ONLY valid JSON:
{{
  "evaluations": [
    {{
      "thread_id": "abc123",
      "is_relevant": true,
      "relevance_score": 8,
      "reasoning": "Brief explanation"
    }},
    ...
  ]
}}
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["research_question", "threads_batch"]
    )

    json_parser = JsonOutputParser()
    chain = prompt | model | json_parser

    relevant_threads = []
    batch_size = 10

    for batch_start in range(0, len(all_threads), batch_size):
        batch_end = min(batch_start + batch_size, len(all_threads))
        batch = all_threads[batch_start:batch_end]

        # Format threads for LLM
        threads_text = ""
        for i, thread in enumerate(batch, 1):
            threads_text += f"\n{i}. ID: {thread['id']}\n"
            threads_text += f"   Title: {thread['title']}\n"
            threads_text += f"   Content: {thread['selftext'][:200]}...\n"
            threads_text += f"   Score: {thread['score']}, Comments: {thread['num_comments']}\n"

        try:
            result = chain.invoke({
                "research_question": state["research_question"],
                "threads_batch": threads_text
            })

            evaluations = result.get("evaluations", [])

            # Map evaluations back to threads
            for eval_data in evaluations:
                thread_id = eval_data.get("thread_id")
                is_relevant = eval_data.get("is_relevant", False)
                relevance_score = eval_data.get("relevance_score", 0)
                reasoning = eval_data.get("reasoning", "")

                # Find the corresponding thread
                thread = next((t for t in batch if t["id"] == thread_id), None)
                if thread and is_relevant and relevance_score >= 6:
                    thread["relevance_score"] = relevance_score
                    thread["relevance_reasoning"] = reasoning
                    relevant_threads.append(thread)
                    logger.info(f"✓ Relevant [{relevance_score}/10]: {thread['title'][:60]}...")

        except Exception as e:
            logger.error(f"Error filtering batch: {e}")
            # If filtering fails, include threads by default
            for thread in batch:
                thread["relevance_score"] = 6
                thread["relevance_reasoning"] = "Batch filtering failed, included by default"
                relevant_threads.append(thread)

    state["relevant_threads"] = relevant_threads
    logger.info(f"Filtered to {len(relevant_threads)} relevant threads (from {len(all_threads)})")

    return state


def analyze_threads_node(state: ResearchState) -> ResearchState:
    """
    Node 4: Analyze each relevant thread in detail
    """
    logger.info("=== STEP 4: Analyzing Threads ===")

    fetcher = RedditFetcher()
    thread_analyses = []

    for i, thread in enumerate(state["relevant_threads"]):
        logger.info(f"Analyzing thread {i+1}/{len(state['relevant_threads'])}: {thread['title'][:60]}...")

        try:
            # Fetch comments
            comments = fetcher.fetch_post_comments(
                post_id=thread["id"],
                max_comments=50,
                min_score=1
            )

            # Analyze this thread
            analysis = analyze_single_thread(
                thread=thread,
                comments=comments,
                research_question=state["research_question"]
            )

            thread_analyses.append(analysis)

        except Exception as e:
            logger.error(f"Error analyzing thread {thread['id']}: {e}")
            continue

    state["thread_analyses"] = thread_analyses
    logger.info(f"Completed analysis of {len(thread_analyses)} threads")

    return state


def analyze_single_thread(thread: Dict, comments: List[Dict], research_question: str) -> Dict:
    """
    Analyze a single thread with its comments
    """
    # Prepare comments text
    comments_text = "\n\n".join([
        f"[{c['score']} upvotes] {c['author']}: {c['body'][:300]}"
        for c in comments[:30]
    ])

    prompt_template = """
You are analyzing a Reddit discussion to answer a research question.

Research Question: {research_question}

Thread Title: {title}
Thread Content: {selftext}
Thread Score: {score} upvotes, {num_comments} comments

Top Comments:
{comments_text}

Analyze this discussion and extract insights relevant to the research question.

Return ONLY valid JSON:
{{
  "key_insights": [
    "Insight 1",
    "Insight 2"
  ],
  "sentiment": "positive|negative|neutral|mixed",
  "themes": ["theme1", "theme2"],
  "important_quotes": [
    {{
      "quote": "relevant quote text",
      "author": "username",
      "upvotes": 42
    }}
  ]
}}
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["research_question", "title", "selftext", "score", "num_comments", "comments_text"]
    )

    json_parser = JsonOutputParser()
    chain = prompt | model | json_parser

    result = chain.invoke({
        "research_question": research_question,
        "title": thread["title"],
        "selftext": thread["selftext"][:1000],
        "score": thread["score"],
        "num_comments": thread["num_comments"],
        "comments_text": comments_text
    })

    return {
        "thread_id": thread["id"],
        "thread_title": thread["title"],
        "thread_url": thread["url"],
        "relevance_score": thread.get("relevance_score", 0),
        "num_comments_analyzed": len(comments),
        "analysis": result
    }


def synthesize_findings_node(state: ResearchState) -> ResearchState:
    """
    Node 5: Synthesize findings across all analyzed threads
    """
    logger.info("=== STEP 5: Synthesizing Findings ===")

    # Prepare synthesis input
    thread_summaries = []
    for analysis in state["thread_analyses"]:
        thread_summaries.append({
            "title": analysis["thread_title"],
            "insights": analysis["analysis"].get("key_insights", []),
            "sentiment": analysis["analysis"].get("sentiment", "neutral"),
            "themes": analysis["analysis"].get("themes", [])
        })

    prompt_template = """
You are synthesizing research findings from multiple Reddit discussions.

Research Question: {research_question}
Subreddit: r/{subreddit}
Threads Analyzed: {num_threads}

Thread Summaries:
{thread_summaries}

Synthesize these findings to answer the research question. Identify:
1. Overall sentiment/reaction
2. Common themes across discussions
3. Key insights (with confidence based on evidence)
4. Potential concerns or objections
5. Areas of consensus vs disagreement

Return ONLY valid JSON:
{{
  "overall_sentiment": "positive|negative|neutral|mixed",
  "confidence": "high|medium|low",
  "key_findings": [
    {{
      "finding": "Key insight",
      "evidence_strength": "high|medium|low",
      "appears_in_threads": 5
    }}
  ],
  "common_themes": ["theme1", "theme2"],
  "concerns": ["concern1", "concern2"],
  "consensus_areas": ["area1", "area2"],
  "disagreement_areas": ["area1", "area2"]
}}
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["research_question", "subreddit", "num_threads", "thread_summaries"]
    )

    json_parser = JsonOutputParser()
    chain = prompt | model | json_parser

    result = chain.invoke({
        "research_question": state["research_question"],
        "subreddit": state["subreddit"],
        "num_threads": len(state["thread_analyses"]),
        "thread_summaries": json.dumps(thread_summaries, indent=2)
    })

    state["synthesis"] = result
    logger.info(f"Generated synthesis with {len(result.get('key_findings', []))} key findings")

    return state


def generate_recommendations_node(state: ResearchState) -> ResearchState:
    """
    Node 6: Generate strategic recommendations
    """
    logger.info("=== STEP 6: Generating Recommendations ===")

    prompt_template = """
Based on the research findings, generate strategic recommendations.

Research Question: {research_question}
Synthesis: {synthesis}

Generate 5-10 actionable recommendations with:
- Clear recommendation statement
- Rationale (why this matters)
- Priority level (HIGH/MEDIUM/LOW)
- Supporting evidence from the research

Also identify:
- Risks to be aware of
- Opportunities to pursue

Return ONLY valid JSON:
{{
  "recommendations": [
    {{
      "recommendation": "Action to take",
      "rationale": "Why this matters",
      "priority": "HIGH|MEDIUM|LOW",
      "evidence": "Supporting data"
    }}
  ],
  "risks": [
    {{
      "risk": "Potential risk",
      "mitigation": "How to address it"
    }}
  ],
  "opportunities": ["opportunity1", "opportunity2"]
}}
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["research_question", "synthesis"]
    )

    json_parser = JsonOutputParser()
    chain = prompt | model | json_parser

    result = chain.invoke({
        "research_question": state["research_question"],
        "synthesis": json.dumps(state["synthesis"], indent=2)
    })

    state["recommendations"] = result.get("recommendations", [])
    state["synthesis"]["risks"] = result.get("risks", [])
    state["synthesis"]["opportunities"] = result.get("opportunities", [])

    logger.info(f"Generated {len(state['recommendations'])} recommendations")

    return state


def create_reports_node(state: ResearchState) -> ResearchState:
    """
    Node 7: Create final JSON and Markdown reports
    """
    logger.info("=== STEP 7: Creating Reports ===")

    # Create JSON report
    report_json = {
        "research_question": state["research_question"],
        "subreddit": state["subreddit"],
        "time_period_days": state["time_period_days"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "keywords_generated": state["keywords"],
            "threads_found": len(state["all_threads"]),
            "threads_analyzed": len(state["relevant_threads"]),
            "pass_rate": f"{len(state['relevant_threads'])/len(state['all_threads'])*100:.1f}%" if state["all_threads"] else "0%",
            "total_comments": sum(t.get("num_comments_analyzed", 0) for t in state["thread_analyses"])
        },
        "synthesis": state["synthesis"],
        "recommendations": state["recommendations"],
        "detailed_thread_analyses": state["thread_analyses"]
    }

    state["report_json"] = report_json

    # Create Markdown report
    markdown = create_markdown_report(state)
    state["report_markdown"] = markdown

    logger.info("Reports created successfully")

    return state


def create_markdown_report(state: ResearchState) -> str:
    """
    Generate a human-readable Markdown report
    """
    synthesis = state["synthesis"]

    md = f"""# Reddit Research Report

## Research Question
**{state["research_question"]}**

## Research Parameters
- **Subreddit:** r/{state["subreddit"]}
- **Time Period:** Last {state["time_period_days"]} days
- **Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

---

## How to Read This Report

This report analyzes real Reddit discussions to answer your research question. Here's what each section contains:

- **Executive Summary**: Quick takeaways - start here for the high-level answer
- **Key Findings**: Detailed insights with evidence strength ratings (strong/moderate/weak)
- **Common Themes**: Recurring topics and patterns across discussions
- **Concerns & Objections**: Potential pushback or criticism from the community
- **Opportunities**: Actionable recommendations based on the research
- **Detailed Thread Analysis**: Deep dive into each relevant discussion (with citations)

**Evidence Strength Guide:**
- **Strong**: Multiple high-quality sources with clear consensus
- **Moderate**: Several sources with some agreement
- **Weak**: Limited sources or conflicting information

---

## Research Methodology

### Keywords Generated
"""

    for i, keyword in enumerate(state["keywords"], 1):
        md += f"{i}. \"{keyword}\"\n"

    pass_rate = (len(state["relevant_threads"]) / len(state["all_threads"]) * 100) if state["all_threads"] else 0

    md += f"""
### Data Collection
- **Total threads found:** {len(state["all_threads"])}
- **Threads analyzed:** {len(state["relevant_threads"])}
- **Relevance pass rate:** {pass_rate:.1f}%
- **Total comments analyzed:** {sum(t.get("num_comments_analyzed", 0) for t in state["thread_analyses"])}

---

## Executive Summary

### TL;DR
"""

    # Add top 3 key findings as bullet points
    for finding in synthesis.get("key_findings", [])[:3]:
        md += f"- **{finding.get('finding', 'N/A')}** ({finding.get('evidence_strength', 'N/A')} evidence)\n"

    md += f"""
### Community Response
- **Overall Sentiment:** {synthesis.get("overall_sentiment", "N/A").title()}
- **Confidence Level:** {synthesis.get("confidence", "N/A").title()}
- **Based on:** {len(state["relevant_threads"])} discussions with {sum(t.get("num_comments_analyzed", 0) for t in state["thread_analyses"])} comments analyzed

---

## Key Findings

"""

    for i, finding in enumerate(synthesis.get("key_findings", []), 1):
        md += f"""
### {i}. {finding.get("finding", "N/A")}
- **Evidence Strength:** {finding.get("evidence_strength", "N/A").title()}
- **Mentioned in:** {finding.get("appears_in_threads", 0)} threads
- **Sources:** See [detailed thread analysis](#thread-analysis-details) below

"""

    md += """
---

## Common Themes

"""
    for theme in synthesis.get("common_themes", []):
        md += f"- {theme}\n"

    md += """
---

## Concerns & Objections

"""
    for concern in synthesis.get("concerns", []):
        md += f"- {concern}\n"

    md += """
---

## Recommendations

"""

    for i, rec in enumerate(state["recommendations"], 1):
        md += f"""
### {i}. {rec.get("recommendation", "N/A")} [{rec.get("priority", "N/A").upper()}]

**Rationale:** {rec.get("rationale", "N/A")}

**Evidence:** {rec.get("evidence", "N/A")}

"""

    md += """
---

## Risks

"""
    for risk in synthesis.get("risks", []):
        md += f"""
### {risk.get("risk", "N/A")}
**Mitigation:** {risk.get("mitigation", "N/A")}

"""

    md += """
---

## Opportunities

"""
    for opp in synthesis.get("opportunities", []):
        md += f"- {opp}\n"

    md += """
---

<a name="thread-analysis-details"></a>
## Thread Analysis Details

Each thread below is numbered [1], [2], etc. for easy reference from findings above.

"""

    for idx, analysis in enumerate(state["thread_analyses"], 1):
        thread_id = analysis.get("thread_id", "")
        sentiment = analysis["analysis"].get("sentiment", "N/A")
        relevance = analysis.get("relevance_score", 0)

        md += f"""
<a name="thread-{idx}"></a>
### [{idx}] {analysis.get("thread_title", "N/A")}
[View Thread]({analysis.get("thread_url", "N/A")}) • Sentiment: {sentiment} • Relevance: {relevance}/10

**Key Insights:**
"""
        for insight in analysis["analysis"].get("key_insights", []):
            md += f"- {insight}\n"

        quotes = analysis["analysis"].get("important_quotes", [])[:2]
        if quotes:
            md += f"\n**Notable Comments:**\n"
            for quote in quotes:
                md += f'> "{quote.get("quote", "N/A")}" — u/{quote.get("author", "unknown")}\n\n'

        md += "---\n\n"

    return md


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph Construction - Define the execution flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Create a new graph that uses our State type
graph = StateGraph(ResearchState)

# Add nodes to the graph
graph.add_node("generate_keywords", generate_keywords_node)
graph.add_node("fetch_threads", fetch_threads_node)
graph.add_node("filter_threads", filter_threads_node)
graph.add_node("analyze_threads", analyze_threads_node)
graph.add_node("synthesize_findings", synthesize_findings_node)
graph.add_node("generate_recommendations", generate_recommendations_node)
graph.add_node("create_reports", create_reports_node)

# Define the execution flow
graph.add_edge("generate_keywords", "fetch_threads")
graph.add_edge("fetch_threads", "filter_threads")
graph.add_edge("filter_threads", "analyze_threads")
graph.add_edge("analyze_threads", "synthesize_findings")
graph.add_edge("synthesize_findings", "generate_recommendations")
graph.add_edge("generate_recommendations", "create_reports")
graph.add_edge("create_reports", END)

# Set the starting point
graph.set_entry_point("generate_keywords")

# Compile the graph into an executable pipeline
pipeline = graph.compile()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI Endpoints - Required by AgentSystems platform
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(request: Request, req: InvokeRequest) -> InvokeResponse:
    """
    Main agent endpoint - executes the LangGraph pipeline.
    """
    # Extract the unique thread ID injected by the gateway
    thread_id = request.headers.get("X-Thread-Id", "")

    logger.info(f"Starting research for r/{req.subreddit}: {req.research_question}")

    # Initialize the state with the input data
    initial_state: ResearchState = {
        "subreddit": req.subreddit,
        "research_question": req.research_question,
        "time_period_days": req.time_period_days,
        "keywords": [],
        "all_threads": [],
        "relevant_threads": [],
        "thread_analyses": [],
        "synthesis": {},
        "recommendations": [],
        "report_json": {},
        "report_markdown": ""
    }

    # Execute the graph pipeline
    final_state: ResearchState = pipeline.invoke(initial_state)

    # Generate PDF from markdown
    import base64
    import tempfile

    pdf_base64 = ""
    pdf_bytes = b""
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
            pdf = MarkdownPdf()
            pdf.add_section(Section(final_state["report_markdown"], toc=False))
            pdf.save(tmp_pdf.name)

            # Read PDF and encode as base64
            with open(tmp_pdf.name, 'rb') as f:
                pdf_bytes = f.read()
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

        logger.info("PDF generated successfully")
    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")
        pdf_base64 = ""

    # Save artifacts to /artifacts/{thread_id}/out/
    if thread_id:
        try:
            out_dir = ARTIFACTS_ROOT / thread_id / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Write JSON report
            (out_dir / "report.json").write_text(
                json.dumps(final_state["report_json"], indent=2)
            )

            # Write Markdown report
            (out_dir / "report.md").write_text(final_state["report_markdown"])

            # Write PDF report (if generated successfully)
            if pdf_bytes:
                (out_dir / "report.pdf").write_bytes(pdf_bytes)

            logger.info(f"Artifacts saved to {out_dir}")
        except Exception as e:
            logger.warning(f"Failed to save artifacts: {e}")

    # Return the results
    return InvokeResponse(
        thread_id=thread_id,
        subreddit=final_state["subreddit"],
        research_question=final_state["research_question"],
        report_json=final_state["report_json"],
        report_markdown=final_state["report_markdown"],
        report_pdf_base64=pdf_base64,
        timestamp=datetime.now(timezone.utc)
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "ok", "version": meta.get("version", "0.1.0")}


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    """
    Metadata endpoint.
    """
    return meta
