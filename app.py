import asyncio
from graph import app

async def main():
    while True:
        question = input("\nAsk question: ")
        result = await app.ainvoke({"question": question})
        print("\nAnswer:\n")
        print(result["answer"])

asyncio.run(main())