"""SQL corpus: social-media posts (analyzed_data) normalised for Qdrant + Neo4j, and the SQL ingestion job."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from graphrag.models.documents import JobStatus


class NamedRef(BaseModel):
    """A node a post links to: id is the Neo4j key, name what is shown."""

    id: str
    name: str
    kind: str | None = None  # SocialEntity: person / organisation / location / incident


class PostRecord(BaseModel):
    post_id: str  # "post:<analyzed_data.id>", also the Qdrant chunk_id
    row_id: int
    text: str  # post text, phone numbers masked, cut to SQL_POST_CHARS
    summary: str | None = None  # contextual_understanding: the pipeline's one-paragraph summary
    url: str | None = None
    platform: str  # twitter / facebook / news / whatsapp / youtube
    author: str | None = None  # username
    author_name: str | None = None
    posted_at: datetime | None = None
    language: str = "unknown"
    topic: NamedRef | None = None
    districts: list[NamedRef] = Field(default_factory=list)
    thanas: list[NamedRef] = Field(default_factory=list)
    categories: list[NamedRef] = Field(default_factory=list)
    sub_categories: list[NamedRef] = Field(default_factory=list)
    entities: list[NamedRef] = Field(default_factory=list)  # people, organisations, locations, incidents
    hashtags: list[NamedRef] = Field(default_factory=list)
    mentions: list[NamedRef] = Field(default_factory=list)  # @accounts
    account: NamedRef | None = None  # the author's account
    sentiment: str | None = None
    emotion: str | None = None

    @property
    def date(self) -> str | None:
        return self.posted_at.date().isoformat() if self.posted_at else None

    @property
    def iso_time(self) -> str | None:
        return self.posted_at.strftime("%Y-%m-%dT%H:%M:%SZ") if self.posted_at else None

    def header(self) -> str:
        """One line of facts shown above the post text (in the prompt and the vector)."""
        parts = [self.platform]
        if self.author:
            parts.append(f"@{self.author}")
        if self.date:
            parts.append(self.date)
        if self.districts:
            parts.append("district: " + ", ".join(d.name for d in self.districts))
        if self.thanas:
            parts.append("thana: " + ", ".join(t.name for t in self.thanas))
        cats = [c.name for c in self.categories] + [s.name for s in self.sub_categories]
        if cats:
            parts.append("category: " + " / ".join(dict.fromkeys(cats)))
        if self.sentiment:
            parts.append(f"sentiment: {self.sentiment}")
        return "[" + " | ".join(parts) + "]"

    def document_text(self) -> str:
        lines = [self.header()]
        if self.topic:
            lines.append(f"Topic: {self.topic.name}")
        lines.append(self.text)
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        return "\n".join(lines)


class StanceRecord(BaseModel):
    """sentiment_entities: the stance a post takes towards an entity."""

    post_id: str
    entity: NamedRef
    stance: str
    confidence: float


class Hierarchy(BaseModel):
    """Lookup tables: thana -> district -> commissionerate / range / zone, sub-category -> category."""

    thanas: list[dict[str, str | None]] = Field(default_factory=list)  # id, name, district_id, district, unit_id...
    sub_categories: list[dict[str, str | None]] = Field(default_factory=list)  # id, name, category_id, category, hint
    keyword_aliases: dict[str, list[str]] = Field(default_factory=dict)  # sub-category id -> Hindi/English keywords


class SqlCorpus(BaseModel):
    posts: list[PostRecord]
    stances: list[StanceRecord] = Field(default_factory=list)
    hierarchy: Hierarchy = Field(default_factory=Hierarchy)
    skipped: dict[str, int] = Field(default_factory=dict)  # reason -> rows


class SqlIngestResult(BaseModel):
    source: str
    posts: int
    vectors: int
    topics: int
    accounts: int
    entities: int
    stances: int
    skipped: dict[str, int] = Field(default_factory=dict)
    graph: dict[str, int] = Field(default_factory=dict)  # node/edge counts of the social graph after ingest
    seconds: float


class SqlIngestJob(BaseModel):
    job_id: str
    status: JobStatus = "queued"
    source: str
    kind: Literal["sql"] = "sql"
    stage: str = "queued"
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: SqlIngestResult | None = None
    errors: list[str] = Field(default_factory=list)
