"""Citation values for tests that do not touch the corpus schema."""

import uuid

from pharma_agent.domain.conversation.models import Citation
from pharma_agent.domain.retrieval.models import HydrateStrategy

CURRENT_RELEASE_ID = uuid.UUID("00000000-0000-4000-8000-00000000aa01")
OLD_RELEASE_ID = uuid.UUID("00000000-0000-4000-8000-00000000aa02")
SOURCE_TITLE = "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2)"
SNIPPET = "Người lớn và trẻ em trên 12 tuổi uống 0,5–1 g mỗi 4–6 giờ…"


def chunk_id(n: int) -> uuid.UUID:
    return uuid.UUID(int=0x1000 + n)


def build_citation(
    index: int = 1,
    *,
    chunk: int = 1,
    release_id: uuid.UUID = CURRENT_RELEASE_ID,
    strategy: HydrateStrategy = HydrateStrategy.FULL_SECTION,
    block: tuple[int, ...] = (1,),
) -> Citation:
    return Citation(
        index=index,
        chunk_version_id=chunk_id(chunk),
        release_id=release_id,
        strategy=strategy,
        block_chunk_version_ids=[chunk_id(n) for n in block],
        source=SOURCE_TITLE,
        title="Paracetamol",
        section="Liều lượng và cách dùng",
        start_page=812,
        end_page=813,
        snippet=SNIPPET,
    )
