"""
Chapter 32: Agentic RAG - Autonomous Retrieval & Reasoning
RAG systems with AI agents that plan, retrieve, and reason
Author: Eduard Dulharu (@eduardd76)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# ============================================================================
# Data Models
# ============================================================================

class AgentAction(str, Enum):
    """Agent actions"""
    RETRIEVE = "RETRIEVE"
    REFORMULATE_QUERY = "REFORMULATE_QUERY"
    DECOMPOSE_QUERY = "DECOMPOSE_QUERY"
    SYNTHESIZE = "SYNTHESIZE"
    VERIFY = "VERIFY"
    STOP = "STOP"


class RetrievalSource(str, Enum):
    """Retrieval sources"""
    VECTOR_DB = "VECTOR_DB"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"
    CONFIG_FILES = "CONFIG_FILES"
    LOGS = "LOGS"
    EXTERNAL_API = "EXTERNAL_API"


@dataclass
class AgentStep:
    """Single agent reasoning step"""
    step_number: int
    action: AgentAction
    reasoning: str
    parameters: Dict
    result: Optional[str] = None


@dataclass
class Document:
    """Simple document with content"""
    doc_id: str
    content: str
    metadata: Dict[str, str]
    relevance_score: float = 0.0


@dataclass
class AgentTrace:
    """Complete agent execution trace"""
    query: str
    steps: List[AgentStep]
    final_answer: str
    confidence: float
    sources_used: List[str]


# ============================================================================
# Pydantic Models for LLM Outputs
# ============================================================================

class QueryPlan(BaseModel):
    """Agent's plan for answering query"""
    query_type: str = Field(description="SIMPLE, MULTI_HOP, COMPARISON, or COMPLEX")
    required_sources: List[str] = Field(description="Sources needed (VECTOR_DB, KNOWLEDGE_GRAPH, etc.)")
    sub_queries: List[str] = Field(description="If query needs decomposition (0-3 sub-queries)")
    reasoning: str = Field(description="Why this plan will work (1-2 sentences)")


class NextAction(BaseModel):
    """Agent's next action decision"""
    action: str = Field(description="RETRIEVE, REFORMULATE_QUERY, DECOMPOSE_QUERY, SYNTHESIZE, VERIFY, or STOP")
    reasoning: str = Field(description="Why take this action (1-2 sentences)")
    parameters: Dict[str, str] = Field(description="Parameters for the action")


class QueryReformulation(BaseModel):
    """Reformulated query"""
    original_query: str = Field(description="Original user query")
    reformulated_query: str = Field(description="Improved query for better retrieval")
    changes_made: List[str] = Field(description="What was changed and why (2-3 items)")


class AnswerSynthesis(BaseModel):
    """Synthesized answer from multiple sources"""
    answer: str = Field(description="Complete answer (2-4 sentences)")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW")
    supporting_facts: List[str] = Field(description="Key facts from sources (3-5 items)")
    gaps: List[str] = Field(description="Information gaps or uncertainties (0-2 items)")
    sources_cited: List[str] = Field(description="Source IDs used")


class AnswerVerification(BaseModel):
    """Verification of answer quality"""
    is_correct: bool = Field(description="True if answer appears correct")
    completeness_score: int = Field(description="0-100, how complete is the answer")
    issues_found: List[str] = Field(description="Issues or inconsistencies (0-3 items)")
    needs_more_retrieval: bool = Field(description="True if more information needed")
    additional_queries: List[str] = Field(description="Additional queries if needed (0-2 items)")


# ============================================================================
# Agentic RAG System
# ============================================================================

class AgenticRAGSystem:
    """RAG system with autonomous agents"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=self.api_key,
            temperature=0
        )

        # Simple document store (in production, use vector DB)
        self.documents: Dict[str, Document] = {}
        self.max_iterations = 5

    def add_document(self, doc_id: str, content: str, metadata: Dict[str, str]):
        """Add document to knowledge base"""
        self.documents[doc_id] = Document(
            doc_id=doc_id,
            content=content,
            metadata=metadata
        )

    def plan_query(self, query: str) -> QueryPlan:
        """
        Agent plans how to answer the query

        Args:
            query: User query

        Returns:
            Query plan with strategy
        """
        parser = PydanticOutputParser(pydantic_object=QueryPlan)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query planning agent.
Analyze the query and create a retrieval plan.

Simple: Single fact lookup (use VECTOR_DB)
Multi-hop: Requires multiple retrieval steps (use VECTOR_DB + KNOWLEDGE_GRAPH)
Comparison: Compare two things (decompose into sub-queries)
Complex: Requires reasoning across multiple sources

{format_instructions}"""),
            ("human", """Query: {query}

Create a retrieval plan.""")
        ])

        chain = prompt | self.llm | parser

        plan = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "query": query
        })

        return plan

    def decide_next_action(self, query: str, steps_so_far: List[AgentStep],
                          current_context: str) -> NextAction:
        """
        Agent decides next action

        Args:
            query: Original query
            steps_so_far: Steps taken so far
            current_context: Context gathered so far

        Returns:
            Next action to take
        """
        parser = PydanticOutputParser(pydantic_object=NextAction)

        steps_summary = "\n".join([
            f"Step {s.step_number}: {s.action.value} - {s.reasoning}"
            for s in steps_so_far
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a RAG agent deciding the next action.

Actions:
- RETRIEVE: Get more documents
- REFORMULATE_QUERY: Improve query for better retrieval
- DECOMPOSE_QUERY: Break into sub-queries
- SYNTHESIZE: Combine information into answer
- VERIFY: Check answer quality
- STOP: Done, return answer

{format_instructions}"""),
            ("human", """Original Query: {query}

Steps So Far:
{steps_summary}

Current Context:
{current_context}

Decide next action.""")
        ])

        chain = prompt | self.llm | parser

        action = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "query": query,
            "steps_summary": steps_summary,
            "current_context": current_context[:1000]
        })

        return action

    def reformulate_query(self, original_query: str, reason: str) -> str:
        """
        Reformulate query for better retrieval

        Args:
            original_query: Original query
            reason: Why reformulation is needed

        Returns:
            Reformulated query
        """
        parser = PydanticOutputParser(pydantic_object=QueryReformulation)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query reformulation agent.
Improve queries to be more specific, add technical terms, fix ambiguity.

{format_instructions}"""),
            ("human", """Original Query: {query}
Reason for Reformulation: {reason}

Reformulate to improve retrieval.""")
        ])

        chain = prompt | self.llm | parser

        reformulation = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "query": original_query,
            "reason": reason
        })

        return reformulation.reformulated_query

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Simple keyword-based retrieval (in production, use embeddings)

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Retrieved documents
        """
        # Simple keyword matching (production would use vector similarity)
        query_terms = set(query.lower().split())

        scored_docs = []
        for doc_id, doc in self.documents.items():
            content_terms = set(doc.content.lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                doc.relevance_score = overlap
                scored_docs.append(doc)

        # Sort by score
        scored_docs.sort(key=lambda d: d.relevance_score, reverse=True)

        return scored_docs[:top_k]

    def synthesize_answer(self, query: str, documents: List[Document]) -> Dict:
        """
        Synthesize answer from retrieved documents

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Synthesized answer
        """
        context = "\n\n".join([
            f"[{doc.doc_id}] {doc.content}"
            for doc in documents
        ])

        parser = PydanticOutputParser(pydantic_object=AnswerSynthesis)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an answer synthesis agent.
Combine information from multiple sources into a coherent answer.
Cite sources and acknowledge gaps.

{format_instructions}"""),
            ("human", """Question: {query}

Sources:
{context}

Synthesize a complete answer.""")
        ])

        chain = prompt | self.llm | parser

        synthesis = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "query": query,
            "context": context
        })

        return {
            "answer": synthesis.answer,
            "confidence": synthesis.confidence,
            "supporting_facts": synthesis.supporting_facts,
            "gaps": synthesis.gaps,
            "sources_cited": synthesis.sources_cited
        }

    def verify_answer(self, query: str, answer: str, sources: List[Document]) -> Dict:
        """
        Verify answer quality

        Args:
            query: Original query
            answer: Generated answer
            sources: Sources used

        Returns:
            Verification result
        """
        sources_text = "\n".join([f"- {doc.doc_id}: {doc.content[:200]}..." for doc in sources])

        parser = PydanticOutputParser(pydantic_object=AnswerVerification)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an answer verification agent.
Check if the answer is correct, complete, and well-supported.

{format_instructions}"""),
            ("human", """Question: {query}

Answer: {answer}

Sources Used:
{sources_text}

Verify answer quality.""")
        ])

        chain = prompt | self.llm | parser

        verification = chain.invoke({
            "format_instructions": parser.get_format_instructions(),
            "query": query,
            "answer": answer,
            "sources_text": sources_text
        })

        return {
            "is_correct": verification.is_correct,
            "completeness_score": verification.completeness_score,
            "issues_found": verification.issues_found,
            "needs_more_retrieval": verification.needs_more_retrieval,
            "additional_queries": verification.additional_queries
        }

    def run_agentic_rag(self, query: str) -> AgentTrace:
        """
        Run complete agentic RAG loop

        Args:
            query: User query

        Returns:
            Agent trace with answer
        """
        steps: List[AgentStep] = []
        current_context = ""
        retrieved_docs: List[Document] = []
        final_answer = ""
        iteration = 0

        # Step 1: Plan
        plan = self.plan_query(query)
        steps.append(AgentStep(
            step_number=len(steps) + 1,
            action=AgentAction.RETRIEVE,
            reasoning=plan.reasoning,
            parameters={"query_type": plan.query_type},
            result=f"Plan created: {plan.query_type}"
        ))

        # Agent loop
        while iteration < self.max_iterations:
            iteration += 1

            # Decide next action
            action = self.decide_next_action(query, steps, current_context)

            if action.action == "STOP":
                break

            elif action.action == "RETRIEVE":
                # Retrieve documents
                search_query = action.parameters.get("query", query)
                docs = self.retrieve_documents(search_query, top_k=3)
                retrieved_docs.extend(docs)

                result_text = f"Retrieved {len(docs)} documents"
                current_context += "\n\n".join([doc.content for doc in docs])

                steps.append(AgentStep(
                    step_number=len(steps) + 1,
                    action=AgentAction.RETRIEVE,
                    reasoning=action.reasoning,
                    parameters=action.parameters,
                    result=result_text
                ))

            elif action.action == "REFORMULATE_QUERY":
                # Reformulate query
                new_query = self.reformulate_query(query, action.reasoning)

                steps.append(AgentStep(
                    step_number=len(steps) + 1,
                    action=AgentAction.REFORMULATE_QUERY,
                    reasoning=action.reasoning,
                    parameters={"original": query, "reformulated": new_query},
                    result=f"Reformulated to: {new_query}"
                ))

                query = new_query  # Update query

            elif action.action == "SYNTHESIZE":
                # Synthesize answer
                synthesis = self.synthesize_answer(query, retrieved_docs)
                final_answer = synthesis["answer"]

                steps.append(AgentStep(
                    step_number=len(steps) + 1,
                    action=AgentAction.SYNTHESIZE,
                    reasoning=action.reasoning,
                    parameters={},
                    result=f"Answer generated (confidence: {synthesis['confidence']})"
                ))

            elif action.action == "VERIFY":
                # Verify answer
                verification = self.verify_answer(query, final_answer, retrieved_docs)

                steps.append(AgentStep(
                    step_number=len(steps) + 1,
                    action=AgentAction.VERIFY,
                    reasoning=action.reasoning,
                    parameters={},
                    result=f"Verified (completeness: {verification['completeness_score']}%)"
                ))

                if not verification["needs_more_retrieval"]:
                    break

        # Build trace
        trace = AgentTrace(
            query=query,
            steps=steps,
            final_answer=final_answer or "Unable to generate answer",
            confidence=0.85,
            sources_used=[doc.doc_id for doc in retrieved_docs]
        )

        return trace


# ============================================================================
# Examples
# ============================================================================

def example_1_simple_agentic_rag():
    """Example 1: Simple agentic RAG"""
    print("=" * 80)
    print("Example 1: Simple Agentic RAG")
    print("=" * 80)

    rag = AgenticRAGSystem()

    # Add documents
    rag.add_document(
        "doc:bgp",
        "BGP is used for routing between autonomous systems. Configure iBGP between core routers using neighbor commands.",
        {"topic": "routing"}
    )
    rag.add_document(
        "doc:ospf",
        "OSPF is an interior gateway protocol. Use Area 0 for the backbone and configure on all router interfaces.",
        {"topic": "routing"}
    )
    rag.add_document(
        "doc:vlan",
        "VLANs segment traffic in switched networks. Configure trunk ports between switches using 802.1Q encapsulation.",
        {"topic": "switching"}
    )

    print("\nDocuments added: 3")
    print("\nQuery: What routing protocol should I use between routers?")

    trace = rag.run_agentic_rag("What routing protocol should I use between routers?")

    print(f"\n{'='*80}")
    print(f"AGENT TRACE")
    print(f"{'='*80}")
    print(f"Query: {trace.query}")
    print(f"Steps Taken: {len(trace.steps)}")

    print(f"\nAgent Steps:")
    for step in trace.steps:
        print(f"\n  Step {step.step_number}: {step.action.value}")
        print(f"    Reasoning: {step.reasoning}")
        if step.result:
            print(f"    Result: {step.result}")

    print(f"\nFinal Answer:")
    print(f"  {trace.final_answer}")

    print(f"\nSources Used:")
    for source in trace.sources_used:
        print(f"  - {source}")


def example_2_query_planning():
    """Example 2: Query planning"""
    print("\n" + "=" * 80)
    print("Example 2: Query Planning")
    print("=" * 80)

    rag = AgenticRAGSystem()

    queries = [
        "What is BGP?",
        "How do BGP and OSPF work together in a network?",
        "Compare BGP and OSPF routing protocols"
    ]

    print("\nAnalyzing different query types...\n")

    for query in queries:
        print(f"Query: {query}")
        plan = rag.plan_query(query)

        print(f"  Type: {plan.query_type}")
        print(f"  Sources: {', '.join(plan.required_sources)}")
        if plan.sub_queries:
            print(f"  Sub-queries: {len(plan.sub_queries)}")
            for sq in plan.sub_queries:
                print(f"    - {sq}")
        print(f"  Reasoning: {plan.reasoning}")
        print()


def example_3_query_reformulation():
    """Example 3: Query reformulation"""
    print("\n" + "=" * 80)
    print("Example 3: Query Reformulation")
    print("=" * 80)

    rag = AgenticRAGSystem()

    original_query = "How to make network faster?"
    reason = "Query is too vague and needs technical specificity"

    print(f"Original Query: {original_query}")
    print(f"Reason: {reason}\n")

    reformulated = rag.reformulate_query(original_query, reason)

    print(f"Reformulated Query: {reformulated}")


def example_4_answer_synthesis():
    """Example 4: Answer synthesis from multiple sources"""
    print("\n" + "=" * 80)
    print("Example 4: Answer Synthesis")
    print("=" * 80)

    rag = AgenticRAGSystem()

    # Create sample documents
    docs = [
        Document("doc:1", "BGP uses path vector algorithm and is classless. Best for internet routing.", {"type": "protocol"}),
        Document("doc:2", "OSPF uses link-state algorithm. Best for enterprise internal routing.", {"type": "protocol"}),
        Document("doc:3", "BGP scales to millions of routes. OSPF recommended for networks < 500 routers.", {"type": "scaling"}),
    ]

    query = "Which routing protocol should I use for my network?"

    print(f"Query: {query}")
    print(f"Documents: {len(docs)}\n")

    synthesis = rag.synthesize_answer(query, docs)

    print(f"{'='*80}")
    print(f"SYNTHESIZED ANSWER")
    print(f"{'='*80}")
    print(f"\nAnswer: {synthesis['answer']}")
    print(f"\nConfidence: {synthesis['confidence']}")

    print(f"\nSupporting Facts:")
    for fact in synthesis['supporting_facts']:
        print(f"  - {fact}")

    if synthesis['gaps']:
        print(f"\nInformation Gaps:")
        for gap in synthesis['gaps']:
            print(f"  - {gap}")

    print(f"\nSources Cited: {', '.join(synthesis['sources_cited'])}")


def example_5_answer_verification():
    """Example 5: Answer verification"""
    print("\n" + "=" * 80)
    print("Example 5: Answer Verification")
    print("=" * 80)

    rag = AgenticRAGSystem()

    query = "What is BGP used for?"
    answer = "BGP is used for routing between autonomous systems on the internet."
    sources = [
        Document("doc:bgp", "BGP (Border Gateway Protocol) is the routing protocol of the internet, used between autonomous systems.", {})
    ]

    print(f"Query: {query}")
    print(f"Answer: {answer}\n")

    verification = rag.verify_answer(query, answer, sources)

    print(f"{'='*80}")
    print(f"VERIFICATION RESULT")
    print(f"{'='*80}")
    print(f"\nCorrect: {'✓' if verification['is_correct'] else '✗'}")
    print(f"Completeness: {verification['completeness_score']}%")

    if verification['issues_found']:
        print(f"\nIssues Found:")
        for issue in verification['issues_found']:
            print(f"  - {issue}")

    print(f"\nNeeds More Retrieval: {'Yes' if verification['needs_more_retrieval'] else 'No'}")

    if verification['additional_queries']:
        print(f"\nSuggested Additional Queries:")
        for q in verification['additional_queries']:
            print(f"  - {q}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nChapter 32: Agentic RAG - Autonomous Retrieval")
    print("RAG systems with AI agents that plan and reason")
    print("Author: Eduard Dulharu (@eduardd76)\n")

    # Run all examples
    example_1_simple_agentic_rag()
    example_2_query_planning()
    example_3_query_reformulation()
    example_4_answer_synthesis()
    example_5_answer_verification()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
