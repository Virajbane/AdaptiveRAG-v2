"""
backfill_pages_fully_lost.py

ONE-TIME MIGRATION SCRIPT — run once, then delete.

Purpose:
  `pages_fully_lost` is a new field on document records, used by the
  ingestion completeness gate in eval_rag.py. Documents ingested before
  this field existed don't have it, so the gate can't trust them.

  This script finds every already-processed document missing the field
  and reprocesses it, using the exact same `process_document()` function
  the app itself uses (same doc_id, same "replace not duplicate"
  guarantees) -- it just calls that function directly instead of going
  through the /documents/{doc_id}/retry HTTP endpoint, since that
  endpoint's "only retry failed/stuck docs" rule is a real safety
  property we don't want to weaken for a one-time backfill.

  Deliberately NOT wired into any permanent code path (no new endpoint,
  no new query param, no relaxed guard) -- once every document has the
  field, this script has no further purpose.

Usage:
    python backfill_pages_fully_lost.py            # process all matching docs
    python backfill_pages_fully_lost.py --dry-run   # just list what would run
    python backfill_pages_fully_lost.py --user-id <id>   # limit to one user
"""

import argparse
import asyncio
import os


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="List documents that would be reprocessed, without doing it.")
    parser.add_argument("--user-id", default=None,
                         help="Limit the backfill to a single user_id.")
    args = parser.parse_args()

    from app.db.mongodb.client import connect_to_mongo, get_db
    from app.api.v1.endpoints.documents import process_document

    await connect_to_mongo()
    db = await get_db()

    query = {
        "pages_fully_lost": {"$exists": False},
        "status": {"$in": ["processed", "processed_with_gaps"]},
    }
    if args.user_id:
        query["user_id"] = args.user_id

    docs = await db.documents.find(query).to_list(length=1000)
    print(f"Found {len(docs)} document(s) missing 'pages_fully_lost'.")

    if not docs:
        return

    if args.dry_run:
        for doc in docs:
            print(f"  - {doc['_id']}  user={doc['user_id']}  filename={doc['filename']}  status={doc['status']}")
        print("\n(dry run -- nothing reprocessed)")
        return

    succeeded, skipped, failed = [], [], []

    for doc in docs:
        doc_id = str(doc["_id"])
        filename = doc.get("filename", "<unknown>")
        storage_path = doc.get("storage_path")

        # process_document's success path deletes the file from disk, and
        # anything else could've cleaned it up since -- check first so one
        # missing file doesn't kill the whole batch, matching the same
        # "file no longer available" case /retry itself guards against.
        if not storage_path or not os.path.exists(storage_path):
            print(f"[SKIP] {doc_id} ({filename}): original file not found at "
                  f"'{storage_path}' -- would need re-upload, not backfill.")
            skipped.append(doc_id)
            continue

        print(f"[REPROCESS] {doc_id} ({filename})...")
        try:
            await process_document(
                doc_id=doc_id,
                user_id=doc["user_id"],
                file_path=storage_path,
                file_type=doc["file_type"],
                db=db,
            )
            # Confirm the field actually landed before calling it a success.
            refreshed = await db.documents.find_one({"_id": doc["_id"]})
            if refreshed and "pages_fully_lost" in refreshed:
                print(f"  -> OK, pages_fully_lost={refreshed['pages_fully_lost']}")
                succeeded.append(doc_id)
            else:
                print(f"  -> WARNING: reprocessed but field still missing -- check logs above.")
                failed.append(doc_id)
        except Exception as e:
            print(f"  -> FAILED: {e}")
            failed.append(doc_id)

    print("\n" + "=" * 50)
    print(f"Done. succeeded={len(succeeded)} skipped={len(skipped)} failed={len(failed)}")
    if skipped:
        print(f"Skipped (need re-upload): {skipped}")
    if failed:
        print(f"Failed (check logs above): {failed}")


if __name__ == "__main__":
    asyncio.run(main())