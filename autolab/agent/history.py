"""ConversationHistory — sliding window with cumulative Haiku compression."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Maximum message length when formatting turns for the summarizer
_MAX_CONTENT_CHARS = 2000


def _is_tool_result_message(msg: dict) -> bool:
    """Return True if msg is a user message whose content contains tool_result blocks."""
    content = msg.get("content", "")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def _format_message_for_summary(msg: dict) -> str:
    """Format a single message dict into a readable line for the summarizer."""
    role = msg.get("role", "?")
    content = msg.get("content", "")
    if isinstance(content, list):
        # Anthropic content blocks: TextBlock, ToolUseBlock, tool_result, etc.
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    parts.append(
                        f"[tool_call: {block.get('name')}({json.dumps(block.get('input', {}), default=str)[:200]})]"
                    )
                elif btype == "tool_result":
                    parts.append(f"[tool_result: {str(block.get('content', ''))[:200]}]")
                else:
                    parts.append(str(block)[:200])
            else:
                # Anthropic SDK content block objects
                try:
                    btype = getattr(block, "type", "")
                    if btype == "text":
                        parts.append(getattr(block, "text", ""))
                    elif btype == "tool_use":
                        parts.append(
                            f"[tool_call: {block.name}({json.dumps(dict(block.input), default=str)[:200]})]"
                        )
                    else:
                        parts.append(str(block)[:200])
                except Exception:
                    parts.append(str(block)[:200])
        content = " | ".join(parts)
    elif not isinstance(content, str):
        content = str(content)

    content = content[:_MAX_CONTENT_CHARS]
    return f"{role}: {content}"


def _compress_with_haiku(
    client: Any,
    model: str,
    previous_summary: Optional[str],
    turns_to_compress: list[dict],
) -> str:
    """Call Haiku to produce a cumulative summary of old turns.

    Args:
        client: anthropic.Anthropic client.
        model: Haiku model ID.
        previous_summary: Existing rolling summary (None on first compression).
        turns_to_compress: List of message dicts being evicted from the live window.

    Returns:
        Updated summary string incorporating both previous_summary and turns_to_compress.
    """
    formatted_turns = "\n".join(_format_message_for_summary(m) for m in turns_to_compress)

    parts = ["You are summarizing an autonomous RL training agent's session log.\n"]
    if previous_summary:
        parts.append(
            f"Previous summary (covers all turns before this batch):\n{previous_summary}\n"
        )
    parts.append(f"New turns to incorporate:\n{formatted_turns}\n")
    parts.append(
        "Produce a single updated summary (150-250 words) that covers everything in both "
        "the previous summary (if any) and the new turns. Focus on: which runs were "
        "started/forked/killed, what hyperparameter changes were made, what eval results "
        "were observed, and what decisions were taken and why. "
        "Be factual and terse. Omit tool-call verbosity and filler text."
    )

    prompt = "\n".join(parts)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception:
        logger.exception("ConversationHistory: Haiku compression failed; using truncated fallback")
        # Fallback: just use the formatted turns (truncated)
        fallback = (previous_summary or "") + "\n" + formatted_turns
        return fallback[:1500]


class ConversationHistory:
    """Manages the agent's conversation with a sliding live window and Haiku compression.

    Design:
    - _all_messages: full history, written to conversation.jsonl for post-hoc review.
    - _live_messages: the window passed to the LLM (at most max_live_turns * 2 messages).
    - _summary: rolling Haiku summary of evicted turns; prepended to _live_messages.

    A "turn" is one user message + one assistant response = 2 messages. Setting
    max_live_turns=10 keeps 20 messages in the live window.

    Args:
        max_live_turns: Number of full turns (user+assistant pairs) to keep uncompressed.
        summarizer_model: Haiku model ID for compression calls.
    """

    def __init__(
        self,
        max_live_turns: int = 10,
        summarizer_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._all_messages: list[dict] = []
        self._live_messages: list[dict] = []
        self._summary: Optional[str] = None
        self._max_live = max_live_turns * 2  # user + assistant per turn
        self._summarizer_model = summarizer_model

    def append_user(self, content: str | list) -> None:
        """Append a user message."""
        msg = {"role": "user", "content": content}
        self._all_messages.append(msg)
        self._live_messages.append(msg)

    def append_assistant(self, response: Any) -> None:
        """Append an assistant response. Accepts Anthropic Message objects or dicts."""
        if isinstance(response, dict):
            msg = response
        else:
            # Anthropic SDK Message object: serialize content blocks
            content = []
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    content.append({"type": "text", "text": block.text})
                elif btype == "tool_use":
                    content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": dict(block.input),
                    })
                else:
                    content.append({"type": str(btype), "raw": str(block)})
            msg = {"role": "assistant", "content": content}
        self._all_messages.append(msg)
        self._live_messages.append(msg)

    def append_tool_result(self, tool_use_id: str, content: str) -> None:
        """Append a tool result as a user message (Anthropic tool_result format)."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                }
            ],
        }
        self._all_messages.append(msg)
        self._live_messages.append(msg)

    def maybe_compress(self, client: Any) -> None:
        """Compress old turns via Haiku if the live window exceeds the limit.

        Idempotent — calling it twice with the same state is safe.
        The summary is cumulative: each compression incorporates the previous summary
        so long-range context is never lost.

        Safe eviction: we never leave a tool_result block as the first message in
        the live window. Anthropic's API requires that every tool_result has a
        corresponding tool_use in the immediately preceding assistant message.
        If the naive boundary would split a tool_use/tool_result pair, we advance
        the boundary until the first kept message is not a tool_result.
        """
        if len(self._live_messages) <= self._max_live:
            return

        # Start with the minimal eviction count needed to get under the limit.
        evict_count = len(self._live_messages) - self._max_live

        # Advance boundary forward past any tool_result messages so we never
        # leave an orphaned tool_result at the start of the live window.
        while evict_count < len(self._live_messages) and _is_tool_result_message(
            self._live_messages[evict_count]
        ):
            evict_count += 1

        to_compress = self._live_messages[:evict_count]
        self._live_messages = self._live_messages[evict_count:]

        # Cumulative Haiku compression
        self._summary = _compress_with_haiku(
            client, self._summarizer_model, self._summary, to_compress
        )

        # Prepend summary as first message so the LLM always sees it
        summary_msg = {
            "role": "user",
            "content": f"[Earlier session summary]: {self._summary}",
        }
        self._live_messages = [summary_msg] + self._live_messages
        logger.debug(
            "ConversationHistory: compressed %d messages; live window now %d",
            len(to_compress),
            len(self._live_messages),
        )

    def for_llm(self) -> list[dict]:
        """Return the message list to pass to the LLM."""
        return list(self._live_messages)

    def all_messages(self) -> list[dict]:
        """Return the full untruncated message history (for JSONL logging)."""
        return list(self._all_messages)
