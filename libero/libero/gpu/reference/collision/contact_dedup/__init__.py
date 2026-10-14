"""Device-resident MuJoCo 2.3.7 box-contact deduplication."""

from .dedup import ContactDedupWorkspace, dedup_contacts

__all__ = ("ContactDedupWorkspace", "dedup_contacts")
