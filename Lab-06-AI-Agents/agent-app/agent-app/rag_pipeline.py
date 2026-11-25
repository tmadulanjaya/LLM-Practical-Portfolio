import rag_core as rc


def main() -> None:
    ok, msg = rc.check_ollama()
    if not ok:
        raise SystemExit(msg)

    vs = rc.get_vectorstore()
    if not rc.list_documents(vs):
        raise SystemExit("The index is empty. Run `python ingest.py` first.")

    history: list[dict] = []
    while True:
        q = input("\nAsk a university question (or exit): ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue

        standalone, results = rc.ask(vs, q, history)
        if standalone != q:
            print(f"(searching for: {standalone})")

        print("\nAnswer: ", end="", flush=True)
        answer = ""
        for token in rc.answer_stream(standalone, results):
            print(token, end="", flush=True)
            answer += token
        print()

        if results:
            print("\nSources:")
            for c in rc.source_cards(results):
                print(f"  - {c['label']} (distance {c['distance']})")

        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
