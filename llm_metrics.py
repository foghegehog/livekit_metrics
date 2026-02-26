import asyncio
import aiofiles
import csv
import os

from livekit import agents, rtc
from livekit.agents import AgentServer,AgentSession, Agent, room_io
from livekit.agents import inference
from livekit.plugins import openai, elevenlabs, noise_cancellation, silero, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics, EOUMetrics

from dotenv import load_dotenv
load_dotenv(".env.local")

# Models Available Directly in LiveKit Inference
# openai/gpt‑4o‑mini
# openai/gpt‑4.1‑mini / nano
# openai/gpt‑5‑mini / nano
# openai/gpt‑oss‑120b

# Path to your CSV file
#llm_name = "openai/gpt-4o"
#llm_name = "openai/gpt-4o-mini"
#llm_name = "openai/gpt-4.1"
#llm_name = "openai/gpt-4.1-mini"
#llm_name = "openai/gpt-4.1-nano"
#llm_name = "openai/gpt-5"
#llm_name = "openai/gpt-5-nano"
#llm_name = "openai/gpt-5-mini"
#llm_name = "openai/gpt-oss-120b"
#llm_name = "cerebras/llama-3.3-70b"
#llm_name = "cerebras/llama-3.1-8b"  
#llm_name = "cerebras/qwen-3-32b"
#llm_name = "cerebras/gpt-oss-120b"
#llm_name = "google/gemini-2.5-pro"
#llm_name = "google/gemini-2.5-flash"
#llm_name = "google/gemini-2.5-flash-lite"
#llm_name = "google/gemini-2.0-flash"
#llm_name = "google/gemini-2.0-flash-lite"
#llm_name = "deepseek-ai/deepseek-v3"
#llm_name = "moonshotai/kimi-k2-instruct"
llm_name = "qwen/qwen3-235b-a22b-instruct"

CSV_FILE = f'logs/llm/{llm_name}.csv'

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

class MetricsAgent(Agent):
    def __init__(self) -> None:                
        llm = inference.LLM(model=llm_name)        
        #llm = openai.LLM.with_cerebras(model=llm_name)
        #stt = inference.STT(model="deepgram/nova-3")
        stt = deepgram.STT()
        #tts = elevenlabs.TTS()
        tts = deepgram.TTS()
        silero_vad = silero.VAD.load()
        
        super().__init__(
            instructions="You are a helpful assistant communicating via voice",
            stt=stt,
            llm=llm,
            tts=tts,
            vad=silero_vad,
        )
        self.turn_detection=MultilingualModel()

        def llm_metrics_wrapper(metrics: LLMMetrics):
            asyncio.create_task(self.on_llm_metrics_collected(metrics))
        llm.on("metrics_collected", llm_metrics_wrapper)


    async def on_llm_metrics_collected(self, metrics: LLMMetrics) -> None:
        print("\n--- LLM Metrics ---")
        print(f"Prompt Tokens: {metrics.prompt_tokens}")
        print(f"Completion Tokens: {metrics.completion_tokens}")
        print(f"Tokens per second: {metrics.tokens_per_second:.4f}")
        print(f"TTFT: {metrics.ttft:.4f}s")
        print("------------------\n")

        # Append metrics to CSV file
        try:
            async with aiofiles.open(CSV_FILE, mode="r") as f:
                header_exists = True
        except FileNotFoundError:
            header_exists = False
        
        async with aiofiles.open(CSV_FILE, mode="a", newline='') as f:
            # Write header if needed
            if not header_exists:
                await f.write("prompt_tokens,completion_tokens,tokens_per_second,ttft\n")
            # Write the metrics row
            await f.write(f"{metrics.prompt_tokens},{metrics.completion_tokens},{metrics.tokens_per_second:.4f},{metrics.ttft:.4f}\n")
    

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    metrics_agent = MetricsAgent()
    session = AgentSession(
        stt=metrics_agent.stt,
        llm=metrics_agent.llm,
        tts=metrics_agent.tts,
        vad=metrics_agent.vad,
        turn_detection=metrics_agent.turn_detection,
    )

    await session.start(
        room=ctx.room,
        agent=metrics_agent,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
