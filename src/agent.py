##STOP SAYING ASTERISK
# src/agent.py
import logging
import re

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
from livekit.agents import tts as tts_adapters  # for TTS fallback
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero

from knowledge import BankingKB

logger = logging.getLogger("agent")
load_dotenv(".env.local")

SBI_AGENT_INSTRUCTIONS = """
You are the **SBI Banking Assistant** — a knowledgeable, courteous assistant for
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

class Assistant(Agent):
    def __init__(self) -> None:
        # TTS: prefer OpenAI TTS (gpt-4o-mini-tts, voice "alloy"), fallback to Cartesia (sonic-2, en).
        # OpenAI TTS is known-good for English; Cartesia stays as a backup. :contentReference[oaicite:2]{index=2}
        primary_tts = openai.TTS(model="gpt-4o-mini-tts", voice="alloy")
        fallback_tts = cartesia.TTS(model="sonic-2", language="en")

        super().__init__(
            instructions=SBI_AGENT_INSTRUCTIONS,
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.3),
            stt=deepgram.STT(model="nova-3", language="en-IN", detect_language=False),
            tts=tts_adapters.FallbackAdapter(tts=[primary_tts, fallback_tts]),
            vad=silero.VAD.load(),
        )

    @function_tool
    async def query_banking_guide(self, context: RunContext, question: str) -> str:
        return kb.query(question)

    @function_tool
    async def search_web(self, context: RunContext, query: str) -> str:
        return kb.web_search(f"{query} site:sbi.co.in OR site:rbi.org.in OR site:npci.org.in")

    # Strip markdown/bullets before TTS so it won't say "asterisk asterisk ..."
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
        room_input_options=RoomInputOptions(
            # For console keep BVC; for telephony switch to BVCTelephony if you notice artifacts.
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    session.generate_reply(instructions=WELCOME_TURN)
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


