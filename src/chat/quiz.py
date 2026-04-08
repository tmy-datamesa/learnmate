"""Test mode — quiz generation and answer evaluation.

Generates questions from wiki documents, evaluates user answers,
and tracks results for a session summary.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.chat.retriever import get_vectorstore
from src.utils.config import CHAT_MODEL, OPENAI_API_KEY

QUESTION_PROMPT = """\
You are a quiz generator for an AI/ML learning system. You have access to \
the user's wiki notes below.

Generate exactly one quiz question based on the provided context. Rules:
- Write the question in Turkish. Use technical terms in English as-is.
- Focus on CONCEPTS, not people. Never ask "what does person X think" or \
"compare person X and Y's views".
- NEVER ask flat definition or listing questions like "X nedir?" or \
"X'in bileşenlerini sayın". These test memorization, not understanding.
- ALWAYS wrap the question in a realistic scenario. The user should feel \
like they're solving a real problem, not taking a textbook exam.
- Try to combine 2+ concepts from the context into one question.
- Good question examples:
  "Bir e-ticaret sitesi için chatbot yapıyorsun. Ürün kataloğu 100.000 \
  item. Kullanıcı 'kırmızı beden M elbise var mı?' diye soruyor. Bu \
  sistemi RAG ile mi kurarsın, long context ile mi? Neden?"
  "Bir Claude Code session'ında 40 mesaj yazdın ve Claude artık önceki \
  talimatlarını unutuyor. Bu durumun teknik adı ne, kök sebebi nedir \
  ve nasıl çözersin?"
  "LearnMate gibi bir RAG chatbot'unda retrieval sonuçları kötüyse, \
  sorun chunking'de mi, embedding'de mi, yoksa query'de mi? Nasıl \
  ayırt edersin?"
- Bad question examples:
  "RAG'ın bileşenlerini sayınız." (listing)
  "Context engineering nedir?" (flat definition)
  "Karpathy ne düşünüyor?" (person-focused)
- Do not include the answer in your response.
- Output ONLY the question, nothing else."""

EVALUATE_PROMPT = """\
You are an answer evaluator for an AI/ML learning quiz. Compare the user's \
answer against the source material below.

Evaluate the answer and respond in this exact format:
Score: [dogru/kismi/yanlis]
Degerlendirme: [evaluation in Turkish — structure below]
Dogru cevap: [The complete correct answer based on wiki sources, in Turkish]

EVALUATION STRUCTURE — follow this order:
1. First, acknowledge what the user got RIGHT. Be specific about which \
concepts or reasoning were correct.
2. Then, note what was MISSING or could be improved.
3. If the user used different words but captured the right idea, that \
counts as correct. Do NOT penalize paraphrasing.

Rules:
- Answer in Turkish. Use technical terms in English as-is.
- "dogru" = the user understands the core concept, even if they used \
different terminology or phrasing than the source. They don't need to \
hit every detail — grasping the key insight is enough.
- "kismi" = some correct reasoning but missing an important dimension \
or trade-off that changes the answer meaningfully.
- "yanlis" = fundamentally wrong reasoning, not just missing details.
- If the user gives a creative or unexpected approach that is still \
technically valid, acknowledge it positively even if it differs from \
the source material.
- Be encouraging. This is a learning tool, not an exam. The goal is \
to help the user build understanding, not to catch them being wrong."""


class TestSession:
    """A quiz session that generates questions and evaluates answers.

    Tracks results across multiple questions for a session summary.
    """

    def __init__(self) -> None:
        """Initialize a new test session."""
        self._vectorstore = get_vectorstore()
        self._llm = ChatOpenAI(
            model=CHAT_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.7,  # Higher for question variety
        )
        self._eval_llm = ChatOpenAI(
            model=CHAT_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.2,  # Lower for consistent evaluation
        )
        self._results: list[dict] = []
        self._current_question: str = ""
        self._current_context: str = ""
        self._current_sources: list[str] = []

    def generate_question(self, topic: str = "") -> tuple[str, list[str]]:
        """Generate a quiz question from wiki content.

        Args:
            topic: Optional topic to focus on. If empty, picks from
                the full wiki.

        Returns:
            Tuple of (question string, list of source names used).
        """
        # Retrieve relevant docs for the topic
        query = topic if topic else "AI ML concepts"
        scored_docs = self._vectorstore.similarity_search_with_relevance_scores(
            query, k=3
        )

        # Format context — skip entity docs (quiz focuses on concepts)
        parts = []
        sources = []
        for doc, score in scored_docs:
            if score < 0.30:
                continue
            doc_type = doc.metadata.get("type", "unknown")
            if doc_type == "entity":
                continue  # Skip people/org docs — quiz is concept-focused
            source = doc.metadata.get("filename", "unknown")
            parts.append(f"[{doc_type}: {source}]\n{doc.page_content}")
            sources.append(f"{doc_type}: {source}")

        context = "\n\n---\n\n".join(parts)

        # Generate question
        messages = [
            SystemMessage(content=QUESTION_PROMPT),
            HumanMessage(content=f"Wiki context:\n{context}"),
        ]
        response = self._llm.invoke(messages)

        self._current_question = response.content
        self._current_context = context
        self._current_sources = sources

        return response.content, sources

    def evaluate_answer(self, user_answer: str) -> dict:
        """Evaluate the user's answer against wiki sources.

        Args:
            user_answer: The user's answer to the current question.

        Returns:
            Dict with keys: question, answer, score, evaluation,
            correct_answer, sources.
        """
        messages = [
            SystemMessage(content=EVALUATE_PROMPT),
            HumanMessage(
                content=(
                    f"Wiki context:\n{self._current_context}\n\n"
                    f"Question: {self._current_question}\n\n"
                    f"User's answer: {user_answer}"
                )
            ),
        ]
        response = self._eval_llm.invoke(messages)

        # Parse the evaluation response
        result = {
            "question": self._current_question,
            "answer": user_answer,
            "evaluation_text": response.content,
            "sources": self._current_sources,
        }

        # Extract score from response
        for line in response.content.splitlines():
            if line.startswith("Score:"):
                score = line.split(":", 1)[1].strip().lower()
                result["score"] = score
                break
        else:
            result["score"] = "unknown"

        self._results.append(result)
        return result

    def get_summary(self) -> str:
        """Generate a session summary of all quiz results.

        Returns:
            Formatted summary string with scores and recommendations.
        """
        if not self._results:
            return "No questions answered yet."

        total = len(self._results)
        correct = sum(1 for r in self._results if r["score"] == "dogru")
        partial = sum(1 for r in self._results if r["score"] == "kismi")
        wrong = sum(1 for r in self._results if r["score"] == "yanlis")

        lines = [
            f"\n{'=' * 50}",
            f"SESSION SUMMARY — {total} questions",
            f"{'=' * 50}",
            f"  Dogru:  {correct}",
            f"  Kismi:  {partial}",
            f"  Yanlis: {wrong}",
        ]

        # List topics that need review
        review_topics = []
        for r in self._results:
            if r["score"] in ("kismi", "yanlis"):
                review_topics.extend(r["sources"])

        if review_topics:
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for t in review_topics:
                if t not in seen:
                    seen.add(t)
                    unique.append(t)
            lines.append("\nTekrar calis:")
            for t in unique:
                lines.append(f"  - {t}")

        return "\n".join(lines)
