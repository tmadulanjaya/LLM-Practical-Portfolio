import argparse
import shutil

import rag_core as rc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete the existing index first")
    args = parser.parse_args()

    ok, msg = rc.check_ollama()
    if not ok:
        raise SystemExit(msg)

    if args.reset:
        shutil.rmtree(rc.CHROMA_DIR, ignore_errors=True)
        print("Existing index deleted.")

    vs = rc.get_vectorstore()
    rc.DATA_DIR.mkdir(exist_ok=True)
    if not list(rc.DATA_DIR.glob("*.pdf")):
        raise SystemExit(f"No PDFs found. Put your files in ./{rc.DATA_DIR}/ and run again.")

    for name, status in rc.ingest_folder(vs).items():
        print(f"  {name}: {status}")

    print("\nIndexed documents:")
    for name, n in rc.list_documents(vs).items():
        print(f"  {name}: {n} chunks")


if __name__ == "__main__":
    main()
