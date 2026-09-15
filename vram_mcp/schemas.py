"""Stable output shapes for coordination MCP tools.

These TypedDicts are intentionally composed of JSON-compatible values.  The
FastMCP v1 runtime turns the return annotations into output schemas and keeps
the same fields present for successful, refused, failed, and unknown results.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


Outcome = Literal["succeeded", "refused", "failed", "unknown"]
OperationLifecycle = Literal["in_flight", "unknown"]


class ClaimEntry(TypedDict):
    claim_id: str
    kind: Literal["model", "reservation"]
    model: str | None
    gb: float | None
    pid: int | None
    owner: str
    purpose: str
    claimed_at: str
    renewed_at: str
    ttl_seconds: int
    expires_at: str


class OperationEntry(TypedDict):
    operation_id: str
    model: str
    kind: str
    scope: str
    started_at: str
    expires_at: str
    pending_until: str
    lifecycle: OperationLifecycle
    owner_live: bool
    lease_expired: bool
    outcome: Literal["unknown"] | None
    reason: str | None
    retry_count: int
    retry_after: str | None


class CoordinationBase(TypedDict):
    ok: bool
    outcome: Outcome
    model: str | None
    reason: str | None
    detail: str | None
    summary: str


class ListClaimsResult(TypedDict):
    ok: bool
    outcome: Outcome
    claims: list[ClaimEntry]
    operations: list[OperationEntry]
    summary: str


class ClaimResult(CoordinationBase):
    claim_id: str | None
    expires_at: str | None


class ReserveResult(CoordinationBase):
    claim_id: str | None
    expires_at: str | None
    gb: float | None


class RenewResult(CoordinationBase):
    claim_id: str | None
    expires_at: str | None


class ReleaseResult(CoordinationBase):
    claim_id: str | None


class ResidencyResult(CoordinationBase):
    operation_id: str | None
    pending_until: str | None
    retry_after: str | None
    retry_count: int
    keep_alive: str | None
    resident: bool | None
    protected: bool | None
    busy: bool | None
    claims: list[ClaimEntry] | None
    operations: list[OperationEntry] | None
    observations: dict[str, Any] | None
    reservations: list[ClaimEntry] | None
    refused: bool | None
    size_verified: bool | None
    model_size_mb: int | None
    headroom_mb: int | None
    reserved_mb: int | None
    additional_mb: int | None
    scope: str | None
    free_mb: int | None
    coordination_warning: str | None


class EnsureFreeResult(CoordinationBase):
    operation_id: str | None
    pending_until: str | None
    retry_after: str | None
    retry_count: int
    already_free: bool
    free_mb: int | None
    unloaded: list[str]
    declined: list[dict[str, Any]]
    attempts: list[ResidencyResult]
    target_mb: int | None
    reserved_mb: int | None
    observations: dict[str, Any]
