from __future__ import annotations

import argparse
import asyncio
from datetime import date
from pathlib import Path

from app.litcal import LitCalClient, import_general_roman_year
from app.main import default_content_root
from app.store import ReleaseStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage versioned Catholic Companion content")
    parser.add_argument("--content-root", type=Path, default=default_content_root())
    commands = parser.add_subparsers(dest="command", required=True)

    importer = commands.add_parser("import-year", help="Import a draft General Roman calendar")
    importer.add_argument("year", type=int)

    publisher = commands.add_parser("publish", help="Publish a reviewed draft")
    publisher.add_argument("release_id")
    publisher.add_argument("--reviewer", required=True)
    publisher.add_argument("--reviewed-at", required=True, type=date.fromisoformat)
    publisher.add_argument("--permission-note", required=True)

    exporter = commands.add_parser(
        "export-android", help="Export a published CalendarBundle for Android assets"
    )
    exporter.add_argument("release_id")
    exporter.add_argument("output", type=Path)
    return parser


async def import_year(store: ReleaseStore, year: int) -> None:
    payload = await LitCalClient().fetch_general_roman_year(year)
    release = import_general_roman_year(payload, year)
    path = store.save_draft(release)
    print(f"Imported draft {release.release_id}")
    print(f"Days: {len(release.bundle.days)}")
    print(f"Celebrations: {len(release.bundle.celebrations)}")
    print(f"Reading references: {len(release.bundle.readings)}")
    print(f"Saved: {path}")


def main() -> None:
    args = build_parser().parse_args()
    store = ReleaseStore(args.content_root)
    if args.command == "import-year":
        asyncio.run(import_year(store, args.year))
    elif args.command == "publish":
        path = store.publish(
            release_id=args.release_id,
            reviewer=args.reviewer,
            reviewed_at=args.reviewed_at,
            permission_note=args.permission_note,
        )
        print(f"Published: {path}")
    elif args.command == "export-android":
        release = store.get_published(args.release_id)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            release.bundle.model_dump_json(by_alias=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Exported Android bundle: {args.output}")


if __name__ == "__main__":
    main()
