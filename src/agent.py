# ## working on lower latency, better math calucations, more accurate banking queries, up to date knowledge base
# # src/agent.py
# import logging
# import re
# import asyncio
# import contextlib
# from datetime import datetime
# from zoneinfo import ZoneInfo

# from dotenv import load_dotenv
# from livekit.agents import (
#     NOT_GIVEN,
#     Agent,
#     AgentFalseInterruptionEvent,
#     AgentSession,
#     JobContext,
#     MetricsCollectedEvent,
#     RoomInputOptions,
#     RunContext,
#     WorkerOptions,
#     cli,
#     metrics,
# )
# from livekit.agents.llm import function_tool
# from livekit.agents import tts as tts_adapters
# from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero

# from knowledge import BankingKB
# from calc_tools import FDMaturity, TDSEstimate, premature_effective_rate

# logger = logging.getLogger("agent")
# load_dotenv(".env.local")

# SBI_AGENT_INSTRUCTIONS = """
# You are the SBI Banking Assistant — a knowledgeable, courteous assistant for
# State Bank of India (SBI). You help users understand SBI products, fees, interest
# rates, processes (YONO/branch/Internet Banking), and applicable RBI/NPCI rules.

# Safety & privacy (strict):
# - Never ask for/accept OTP, PIN, CVV, full card number, passwords, or full account number.
# - Do not collect personal data beyond first name and general context.
# - For account-specific ops (card block, dispute, mobile change), provide the official steps
#   (SBI helpline/YONO/branch/netbanking menu) and STOP; do not collect sensitive info.

# Answering rules:
# 1) Prefer the local knowledge base (SBI/RBI/NPCI documents).
# 2) If not found or time-sensitive, do a web check of official pages and say “as of <date>”.
# 3) Keep replies short (voice-friendly), clear, and polite. Mirror the caller’s language (English/Hindi).
# 4) If details vary by account/product, say so and give the simple way to verify.
# 5) Never use Markdown or special symbols in responses; speak plain sentences. For lists, use 1. 2. 3. only.

# When the call/console session starts, greet briefly and ask how you can help.
# """

# WELCOME_TURN = "Hello! I’m the SBI Banking Assistant. How can I help with your banking question today?"

# # one global instance (lazy FAISS load)
# kb = BankingKB()


# def _normalize_voice_text(text: str) -> str:
#     # convert common bullets to plain
#     text = text.replace("•", "- ").replace("·", "- ")
#     # strip markdown-ish formatting
#     text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
#     text = re.sub(r"\*(.*?)\*", r"\1", text)
#     text = re.sub(r"_([^_]+)_", r"\1", text)
#     text = re.sub(r"`([^`]+)`", r"\1", text)
#     # collapse decorative symbols
#     text = re.sub(r"[#>~=^]+", "", text)
#     # tidy spaces
#     text = re.sub(r"\s{2,}", " ", text)
#     return text.strip()


# async def _bounded(coro, timeout: float = 2.0, fallback: str = "Sorry, I couldn’t fetch that fast enough."):
#     with contextlib.suppress(asyncio.TimeoutError):
#         return await asyncio.wait_for(coro, timeout=timeout)
#     return fallback


# class Assistant(Agent):
#     def __init__(self) -> None:
#         llm = openai.LLM(
#             model="gpt-4o-mini",
#             temperature=0.2,        # fine
#             # NOTE: do NOT pass max_output_tokens / presence_penalty here
#         )

#         super().__init__(
#             instructions=SBI_AGENT_INSTRUCTIONS,
#             llm=llm,
#             stt=deepgram.STT(model="nova-3", language="en-IN", detect_language=False),
#             tts=tts_adapters.FallbackAdapter(tts=[
#                 openai.TTS(model="gpt-4o-mini-tts", voice="alloy"),
#                 cartesia.TTS(model="sonic-2", language="en"),
#             ]),
#             vad=silero.VAD.load(),
#         )

#     # --------- tools (RAG, web, time, math) ---------
#     @function_tool
#     async def query_banking_guide(self, context: RunContext, question: str) -> str:
#         """Search the SBI/RBI/NPCI local knowledge base and return relevant excerpts."""
#         return kb.query(question)

#     @function_tool
#     async def search_web(self, context: RunContext, query: str) -> str:
#         """Fallback web search (prefers official domains). Bounded to ~2 seconds."""
#         q = f"{query} site:sbi.co.in OR site:rbi.org.in OR site:npci.org.in"
#         return await _bounded(kb.web_search_async(q), 2.0)

#     @function_tool
#     async def current_time(self, context: RunContext, tz: str = "Asia/Kolkata") -> str:
#         """Return current time formatted for voice."""
#         try:
#             now = datetime.now(ZoneInfo(tz))
#         except Exception:
#             now = datetime.utcnow()
#         return now.strftime("%b %d, %Y, %I:%M %p")

#     @function_tool
#     async def fd_maturity(
#         self,
#         context: RunContext,
#         principal: float,
#         annual_rate_pct: float,
#         years: float,
#         compounding: str = "quarterly",
#     ):
#         """Compute FD maturity assuming compound interest (default quarterly)."""
#         return FDMaturity(principal, annual_rate_pct, years, compounding).maturity()

#     @function_tool
#     async def fd_premature_effective_rate(
#         self,
#         context: RunContext,
#         applicable_rate_pct: float,
#         contracted_rate_pct: float,
#         deposit_amount: float,
#     ):
#         """Apply SBI retail premature withdrawal penalty to get an effective annual rate."""
#         return {"effective_rate_pct": premature_effective_rate(applicable_rate_pct, contracted_rate_pct, deposit_amount)}

#     @function_tool
#     async def estimate_tds(
#         self,
#         context: RunContext,
#         annual_interest: float,
#         senior_citizen: bool = False,
#         has_pan: bool = True,
#     ):
#         """Estimate TDS on bank interest u/s 194A (simplified)."""
#         return TDSEstimate(annual_interest, senior_citizen, has_pan).tds()

#     # strip bullets/markdown in streaming so TTS won't speak "asterisk asterisk"
#     async def llm_node(self, chat_ctx, tools, model_settings=None):
#         async with self.llm.chat(chat_ctx=chat_ctx, tools=tools, tool_choice=None) as stream:
#             async for chunk in stream:
#                 if chunk is None:
#                     continue
#                 content = getattr(chunk.delta, "content", None) if hasattr(chunk, "delta") else None
#                 if content is not None:
#                     cleaned = _normalize_voice_text(content)
#                     if cleaned != content:
#                         chunk.delta.content = cleaned
#                 yield chunk


# async def entrypoint(ctx: JobContext):
#     # Log context (handy in Cloud logs)
#     ctx.log_context_fields = {"room": ctx.room.name}

#     # Keep pipeline components on the Agent; AgentSession just orchestrates
#     session = AgentSession(preemptive_generation=True)

#     @session.on("agent_false_interruption")
#     def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
#         logger.info("false positive interruption, resuming")
#         session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

#     usage_collector = metrics.UsageCollector()

#     @session.on("metrics_collected")
#     def _on_metrics_collected(ev: MetricsCollectedEvent):
#         metrics.log_metrics(ev.metrics)
#         usage_collector.collect(ev.metrics)

#     async def log_usage():
#         logger.info(f"Usage: {usage_collector.get_summary()}")

#     ctx.add_shutdown_callback(log_usage)

#     # Start (noise_cancellation=BVC is good for console; switch to BVCTelephony for PSTN if needed)
#     await session.start(
#         agent=Assistant(),
#         room=ctx.room,
#         room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
#     )

#     # Greet once
#     session.generate_reply(instructions=WELCOME_TURN)

#     # Join/connect
#     await ctx.connect()


# if __name__ == "__main__":
#     cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))














## WITH EMAIL FUNCTIONALITY
# src/agent.py
import logging
import re
import asyncio
import contextlib
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents import tts as tts_adapters
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero

from knowledge import BankingKB
from calc_tools import FDMaturity, TDSEstimate, premature_effective_rate
from email_utils import send_email_async  # <-- NEW

logger = logging.getLogger("agent")
load_dotenv(".env.local")

SBI_AGENT_INSTRUCTIONS = """
You are the SBI Banking Assistant — a knowledgeable, courteous assistant for
State Bank of India (SBI). You help users understand SBI products, fees, interest
rates, processes (YONO/branch/Internet Banking), and applicable RBI/NPCI rules.

Safety & privacy (strict):
- Never ask for/accept OTP, PIN, CVV, full card number, passwords, or full account number.
- Do not collect personal data beyond first name and general context.
- For account-specific ops (card block, dispute, mobile change), provide the official steps
  (SBI helpline/YONO/branch/netbanking menu) and STOP; do not collect sensitive info.

Answering rules:
1) Prefer the local knowledge base (SBI/RBI/NPCI documents).
2) If not found or time-sensitive, do a web check of official pages and say “as of <date>”.
3) Keep replies short (voice-friendly), clear, and polite. Mirror the caller’s language (English/Hindi).
4) If details vary by account/product, say so and give the simple way to verify.
5) Never use Markdown or special symbols in responses; speak plain sentences. For lists, use 1. 2. 3. only.
6) If the user asks to email a summary, quote, or steps:
   - If you do not have their email address, ASK for it.
   - Confirm subject and a short preview before sending.
   - Then call the send_email tool.
When the call/console session starts, greet briefly and ask how you can help.
"""

WELCOME_TURN = "Hello! I’m the SBI Banking Assistant. How can I help with your banking question today?"

kb = BankingKB()


def _normalize_voice_text(text: str) -> str:
    text = text.replace("•", "- ").replace("·", "- ")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[#>~=^]+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


async def _bounded(coro, timeout: float = 2.0, fallback: str = "Sorry, I couldn’t fetch that fast enough."):
    with contextlib.suppress(asyncio.TimeoutError):
        return await asyncio.wait_for(coro, timeout=timeout)
    return fallback


def _looks_like_email(s: str) -> bool:
    if not s or "@" not in s or "." not in s:
        return False
    bad = set(" ,;\"'()<>")
    return not any(ch in bad for ch in s)


class Assistant(Agent):
    def __init__(self) -> None:
        llm = openai.LLM(
            model="gpt-4o-mini",
            temperature=0.2,
        )

        super().__init__(
            instructions=SBI_AGENT_INSTRUCTIONS,
            llm=llm,
            stt=deepgram.STT(model="nova-3", language="en-IN", detect_language=False),
            tts=tts_adapters.FallbackAdapter(tts=[
                openai.TTS(model="gpt-4o-mini-tts", voice="alloy"),
                cartesia.TTS(model="sonic-2", language="en"),
            ]),
            vad=silero.VAD.load(),
        )

    # --------- tools (RAG, web, time, math, email) ---------
    @function_tool
    async def query_banking_guide(self, context: RunContext, question: str) -> str:
        """Search the SBI/RBI/NPCI local knowledge base and return relevant excerpts."""
        return kb.query(question)

    @function_tool
    async def search_web(self, context: RunContext, query: str) -> str:
        """Fallback web search (prefers official domains). Bounded to ~2 seconds."""
        q = f"{query} site:sbi.co.in OR site:rbi.org.in OR site:npci.org.in"
        return await _bounded(kb.web_search_async(q), 2.0)

    @function_tool
    async def current_time(self, context: RunContext, tz: str = "Asia/Kolkata") -> str:
        """Return current time formatted for voice."""
        try:
            now = datetime.now(ZoneInfo(tz))
        except Exception:
            now = datetime.utcnow()
        return now.strftime("%b %d, %Y, %I:%M %p")

    @function_tool
    async def fd_maturity(
        self,
        context: RunContext,
        principal: float,
        annual_rate_pct: float,
        years: float,
        compounding: str = "quarterly",
    ):
        """Compute FD maturity assuming compound interest (default quarterly)."""
        return FDMaturity(principal, annual_rate_pct, years, compounding).maturity()

    @function_tool
    async def fd_premature_effective_rate(
        self,
        context: RunContext,
        applicable_rate_pct: float,
        contracted_rate_pct: float,
        deposit_amount: float,
    ):
        """Apply SBI retail premature withdrawal penalty to get an effective annual rate."""
        return {"effective_rate_pct": premature_effective_rate(applicable_rate_pct, contracted_rate_pct, deposit_amount)}

    @function_tool
    async def estimate_tds(
        self,
        context: RunContext,
        annual_interest: float,
        senior_citizen: bool = False,
        has_pan: bool = True,
    ):
        """Estimate TDS on bank interest u/s 194A (simplified)."""
        return TDSEstimate(annual_interest, senior_citizen, has_pan).tds()

    @function_tool
    async def send_email(
        self,
        context: RunContext,
        to_email: str,
        subject: str,
        message: str,
        cc_email: str | None = None,
    ) -> str:
        """Send an email via Gmail SMTP. If to_email is missing, ask the user for it first."""
        if not _looks_like_email(to_email):
            return "I need a valid email address to send this. Please provide an email like name@example.com."
        preview = (message or "").strip()
        if len(preview) > 9000:
            preview = preview[:9000] + "\n\n[truncated]"
        result = await send_email_async(to_email, subject or "(no subject)", preview, cc_email)
        return result

    # strip bullets/markdown in streaming so TTS won't say "asterisk asterisk"
    async def llm_node(self, chat_ctx, tools, model_settings=None):
        async with self.llm.chat(chat_ctx=chat_ctx, tools=tools, tool_choice=None) as stream:
            async for chunk in stream:
                if chunk is None:
                    continue
                content = getattr(chunk.delta, "content", None) if hasattr(chunk, "delta") else None
                if content is not None:
                    cleaned = _normalize_voice_text(content)
                    if cleaned != content:
                        chunk.delta.content = cleaned
                yield chunk


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    session = AgentSession(preemptive_generation=True)

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    session.generate_reply(instructions=WELCOME_TURN)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

