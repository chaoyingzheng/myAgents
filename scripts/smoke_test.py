"""Step-1 smoke test: checks the three CoreWeave/W&B services with one API key.

    python scripts/smoke_test.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from money_math.config import MODEL_ID, WEAVE_PROJECT, inference_client, init_weave  # noqa: E402


async def main():
    if not os.environ.get("WANDB_API_KEY"):
        sys.exit("WANDB_API_KEY missing: copy .env.example to .env and fill it in.")

    print(f"1) Weave init -> {WEAVE_PROJECT}")
    init_weave()

    client = inference_client()
    print("\n2) Serverless Inference models:")
    models = await client.models.list()
    for m in models.data:
        print("   ", m.id)

    print(f"\n3) Chat completion with {MODEL_ID} (traced by Weave):")
    # Reasoning models (e.g. Qwen3.6) spend tokens thinking before visible text, so leave room.
    resp = await client.chat.completions.create(
        model=MODEL_ID, messages=[{"role": "user", "content": "Say hi in 5 words."}], max_tokens=4000
    )
    text = resp.choices[0].message.content
    if not text:
        sys.exit(f"   Empty reply from {MODEL_ID} (finish_reason={resp.choices[0].finish_reason}); check MODEL_ID / credits.")
    print("   ", text)

    print("\n4) Sandbox exec:")
    from money_math.sandbox_tool import SandboxSession

    sb = SandboxSession()
    t0 = time.perf_counter()
    out = await sb.run("import sys; print('hello from', sys.version.split()[0]); print(400000*0.065/12/(1-(1+0.065/12)**-360))")
    print("   cold start: %.1fs, total: %.1fs" % (sb.cold_start_seconds, time.perf_counter() - t0))
    print("   " + out.replace("\n", "\n   "))
    await sb.close()
    print("\nAll good.")


if __name__ == "__main__":
    asyncio.run(main())
