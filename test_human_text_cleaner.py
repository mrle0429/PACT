"""已采样人类文本清洗器的最小测试。"""
from __future__ import annotations

import sys

sys.path.insert(0, "/Volumes/Mac/Project/ob")

from src.human_text_cleaner import HumanTextCleaner


def main() -> None:
    cleaner = HumanTextCleaner()

    record = {
        "id": "owt_00000001",
        "text": (
            "Image 1 of / 13 Caption Close Back to Gallery The player scored twice. "
            "https://example.com/watch?v=1 "
            "Source: https://example.com/photo "
            "The team celebrated with supporters."
        ),
        "sentence_count": 4,
    }
    cleaned = cleaner.clean_record(record)
    assert "Image 1 of / 13 Caption Close Back to Gallery The player scored twice." in cleaned.record["text"]
    assert cleaned.record["sentence_count"] == 2
    assert cleaned.removed_sentences == 1

    record2 = {
        "id": "arxiv_1234.5678",
        "text": (
            "The source code is available at <a href=\"https://example.com\">this repository</a>. "
            "Results improved by 5&#39;s margin."
        ),
        "sentence_count": 2,
    }
    cleaned2 = cleaner.clean_record(record2)
    assert "<a href" not in cleaned2.record["text"]
    assert "&#39;" not in cleaned2.record["text"]
    assert "this repository" in cleaned2.record["text"]
    assert cleaned2.record["sentence_count"] == 2

    record2b = {
        "id": "owt_00000002",
        "text": 'He said, “leave it there.” Revenue’s report stayed unchanged.',
        "sentence_count": 2,
    }
    cleaned2b = cleaner.clean_record(record2b)
    assert cleaned2b.record["text"] == record2b["text"]

    record3 = {
        "id": "xsum_00000001",
        "text": "Take part in our new predictor game. The team won 2-0 at home.",
        "sentence_count": 2,
    }
    cleaned3 = cleaner.clean_record(record3)
    assert cleaned3.record["text"] == "Take part in our new predictor game. The team won 2-0 at home."
    assert cleaned3.record["sentence_count"] == 2

    record4 = {
        "id": "xsum_00000002",
        "text": "The report said reforms were needed. The other accused men are:",
        "sentence_count": 2,
    }
    cleaned4 = cleaner.clean_record(record4)
    assert cleaned4.record["text"] == "The report said reforms were needed."
    assert cleaned4.record["sentence_count"] == 1

    print("✓ human_text_cleaner tests passed")


if __name__ == "__main__":
    main()
