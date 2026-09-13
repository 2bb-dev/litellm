from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Literal, Optional, cast, get_args
from uuid import UUID, uuid4, uuid5

from fastapi import HTTPException
from prisma import Prisma, fields, models
from prisma.types import (
    LiteLLM_SpendLogsCreateInput,
    LiteLLM_TeamTableCreateInput,
    LiteLLM_TeamTableUpdateInput,
    LiteLLM_UserTableCreateInput,
    LiteLLM_UserTableUpdateInput,
    LiteLLM_VerificationTokenUpdateInput,
)
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

import litellm
from litellm.litellm_core_utils.accounting_context import AccountingRequest, accounting_call, accounting_request
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import PrismaClient

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_JSON_VALUE = TypeAdapter(JsonValue)
_PROTOCOL = "team-v4"


def _native_policy_data(data: dict[str, JsonValue], model: type[BaseModel]) -> dict[str, JsonValue]:
    # Use the native schema's JSON types; never infer JSON from scalar contents.
    return {
        name: _JSON_VALUE.validate_json(value)
        if isinstance(value, str)
        and name in model.model_fields
        and (
            model.model_fields[name].annotation is fields.Json
            or fields.Json in get_args(model.model_fields[name].annotation)
        )
        else value
        for name, value in data.items()
    }


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["team-v4"] = _PROTOCOL
    request_id: str
    component_id: str
    receipt_id: str
    basis: Literal["actual", "estimate"]
    total: float
    raw: dict[str, JsonValue]
    daily: dict[str, JsonValue]

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()


class ProducerSeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["team-v4"] = _PROTOCOL
    basis: Literal["seal"] = "seal"
    request_id: str
    component_id: str
    receipt_id: str
    expected: tuple[tuple[str, str], ...]
    failed: bool
    nonexecution: bool

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()


OUTBOX_EVENT = TypeAdapter(Envelope | ProducerSeal)


class AccountingReceiptsPending(RuntimeError):
    pass


class AccountingStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    protocol_version: str = _PROTOCOL
    generation_id: str
    incarnation_id: str
    generation_bound: bool
    accepting: Optional[bool]
    pending_requests: Optional[int]
    status: Literal["accounting_pending", "accounting_complete", "accounting_unknown"]


class PostgresAccounting:
    def __init__(
        self,
        client: PrismaClient,
        settlement_transaction: Optional[Callable[[], AbstractAsyncContextManager[Prisma]]] = None,
        *,
        generation_id: Optional[str] = None,
    ):
        self.client = client
        self.generation_id = str(UUID(generation_id)) if generation_id is not None else str(uuid4())
        self.generation_bound = generation_id is not None
        self.incarnation_id = str(uuid4())
        self._settlement_transaction = settlement_transaction or (lambda: client.db.tx(timeout=timedelta(seconds=60)))

    async def initialize(self) -> None:
        rows = await self.client.db.query_raw('SELECT version FROM "LiteLLM_AccountingProtocol" WHERE id=$1', _PROTOCOL)
        if rows != [{"version": 4}]:
            raise RuntimeError("PostgreSQL accounting requires the explicit team-v4 migration")
        if await self.client.db.query_raw('SELECT id FROM "LiteLLM_AccountingGeneration" WHERE version<>4 LIMIT 1'):
            raise RuntimeError("Mixed accounting generations are unsupported; offline qualification required")
        await self.client.db.query_raw("""SELECT g.id, g.version, g.accepting, r.id, r.key_token, r.user_id, r.team_id, r.dispatched, r.input_cost, r.sealed, r.failed, r.closed, r.expected, c.id, c.actual, c.total, c.liability, c.dispatched, c.nonexecution, h.amount, h.disposition, s.kind, s.entity_id, s.epoch, receipt.basis, receipt.payload_hash, receipt.total
            FROM "LiteLLM_AccountingGeneration" g
            JOIN "LiteLLM_AccountingRequest" r ON r.generation_id=g.id
            JOIN "LiteLLM_AccountingComponent" c ON c.request_id=r.id
            JOIN "LiteLLM_AccountingHold" h ON h.request_id=r.id
            JOIN "LiteLLM_AccountingScope" s ON s.id=h.scope_id
            JOIN "LiteLLM_AccountingReceipt" receipt ON receipt.component_id=c.id WHERE false""")
        async with self.client.db.tx() as tx:
            await tx.execute_raw(
                'INSERT INTO "LiteLLM_AccountingGeneration" (id, version) VALUES ($1, 4) ON CONFLICT (id) DO NOTHING',
                self.generation_id,
            )
            rows = await tx.query_raw(
                'SELECT version FROM "LiteLLM_AccountingGeneration" WHERE id=$1 FOR UPDATE', self.generation_id
            )
            if rows != [{"version": 4}]:
                raise RuntimeError("Incompatible accounting generation protocol")

    async def begin(self) -> AccountingRequest:
        request = AccountingRequest(store=self)
        async with self.client.db.tx() as tx:
            rows = await tx.query_raw(
                'SELECT accepting FROM "LiteLLM_AccountingGeneration" WHERE id=$1 FOR UPDATE', self.generation_id
            )
            if rows != [{"accepting": True}]:
                raise HTTPException(503, "Accounting generation is draining")
            await tx.execute_raw(
                'INSERT INTO "LiteLLM_AccountingRequest" (id, generation_id, expected) VALUES ($1, $2, ARRAY[]::text[])',
                request.request_id,
                self.generation_id,
            )
            await tx.execute_raw(
                'INSERT INTO "LiteLLM_AccountingComponent" (id, request_id) VALUES ($1, $2)',
                request.component_id,
                request.request_id,
            )
        return request

    @staticmethod
    def next_reset(duration: str, now: datetime) -> datetime:
        from litellm.litellm_core_utils.duration_parser import get_next_standardized_reset_time

        if re.fullmatch(r"[1-9][0-9]*(?:s|m|h|d|w|mo)", duration) is None:
            raise HTTPException(400, "Invalid accounting budget duration")
        return get_next_standardized_reset_time(duration, now, "UTC")

    @staticmethod
    async def database_time(tx: Prisma) -> datetime:
        value = (await tx.query_raw("SELECT clock_timestamp() AT TIME ZONE 'UTC' AS now"))[0]["now"]
        return TypeAdapter(datetime).validate_python(value).replace(tzinfo=timezone.utc)

    async def key_state(self, tx: Prisma, token: str) -> dict[str, JsonValue]:
        stable = f"key:{token}:initial"
        await tx.execute_raw(
            "INSERT INTO \"LiteLLM_AccountingScope\" (id, kind, entity_id, epoch) VALUES ($1, 'key', $2, 'initial') ON CONFLICT DO NOTHING",
            stable,
            token,
        )
        await tx.query_raw('SELECT id FROM "LiteLLM_AccountingScope" WHERE id=$1 FOR UPDATE', stable)
        rows = await tx.query_raw(
            'SELECT to_jsonb(k) AS data FROM "LiteLLM_VerificationToken" k WHERE token=$1 FOR UPDATE', token
        )
        if not rows:
            raise HTTPException(401, "PostgreSQL accounting requires a database virtual key")
        key = _JSON_OBJECT.validate_python(rows[0]["data"])
        self.validate_key(key)
        duration = key.get("budget_duration")
        if isinstance(duration, str):
            now = await self.database_time(tx)
            reset = key.get("budget_reset_at")
            if reset is None or datetime.fromisoformat(str(reset)).replace(tzinfo=timezone.utc) <= now:
                boundary = self.next_reset(duration, now)
                spend = key["spend"] if reset is None else 0.0
                await tx.execute_raw(
                    'UPDATE "LiteLLM_VerificationToken" SET spend=$3, budget_reset_at=$2::timestamp WHERE token=$1',
                    token,
                    boundary.isoformat(),
                    spend,
                )
                key = {**key, "spend": spend, "budget_reset_at": boundary.isoformat()}
        return key

    async def user_state(self, tx: Prisma, user_id: str) -> Optional[dict[str, JsonValue]]:
        stable = f"user:{user_id}:initial"
        await tx.execute_raw(
            "INSERT INTO \"LiteLLM_AccountingScope\" (id, kind, entity_id, epoch) VALUES ($1, 'user', $2, 'initial') ON CONFLICT DO NOTHING",
            stable,
            user_id,
        )
        await tx.query_raw('SELECT id FROM "LiteLLM_AccountingScope" WHERE id=$1 FOR UPDATE', stable)
        rows = await tx.query_raw(
            "SELECT to_jsonb(u) || jsonb_build_object('organization_memberships', "
            "(SELECT COALESCE(jsonb_agg(m), '[]'::jsonb) FROM \"LiteLLM_OrganizationMembership\" m WHERE m.user_id=u.user_id)) "
            'AS data FROM "LiteLLM_UserTable" u WHERE user_id=$1 FOR UPDATE',
            user_id,
        )
        if not rows:
            return None
        user = _JSON_OBJECT.validate_python(rows[0]["data"])
        duration = user.get("budget_duration")
        if isinstance(duration, str):
            now = await self.database_time(tx)
            reset = user.get("budget_reset_at")
            if reset is None or datetime.fromisoformat(str(reset)).replace(tzinfo=timezone.utc) <= now:
                boundary = self.next_reset(duration, now)
                spend = user["spend"] if reset is None else 0.0
                await tx.execute_raw(
                    'UPDATE "LiteLLM_UserTable" SET spend=$3, budget_reset_at=$2::timestamp WHERE user_id=$1',
                    user_id,
                    boundary.isoformat(),
                    spend,
                )
                user = {**user, "spend": spend, "budget_reset_at": boundary.isoformat()}
        return user

    async def team_state(self, tx: Prisma, team_id: str) -> dict[str, JsonValue]:
        stable = f"team:{team_id}:initial"
        await tx.execute_raw(
            "INSERT INTO \"LiteLLM_AccountingScope\" (id, kind, entity_id, epoch) VALUES ($1, 'team', $2, 'initial') ON CONFLICT DO NOTHING",
            stable,
            team_id,
        )
        await tx.query_raw('SELECT id FROM "LiteLLM_AccountingScope" WHERE id=$1 FOR UPDATE', stable)
        rows = await tx.query_raw(
            'SELECT to_jsonb(t) AS data FROM "LiteLLM_TeamTable" t WHERE team_id=$1 FOR UPDATE', team_id
        )
        if not rows:
            raise HTTPException(404, "PostgreSQL accounting: native Team missing")
        team = _JSON_OBJECT.validate_python(rows[0]["data"])
        duration = team.get("budget_duration")
        if isinstance(duration, str):
            now = await self.database_time(tx)
            reset = team.get("budget_reset_at")
            if reset is None or datetime.fromisoformat(str(reset)).replace(tzinfo=timezone.utc) <= now:
                boundary = self.next_reset(duration, now)
                spend = team["spend"] if reset is None else 0.0
                await tx.execute_raw(
                    'UPDATE "LiteLLM_TeamTable" SET spend=$3, budget_reset_at=$2::timestamp WHERE team_id=$1',
                    team_id,
                    boundary.isoformat(),
                    spend,
                )
                team = {**team, "spend": spend, "budget_reset_at": boundary.isoformat()}
        return team

    async def member_state(self, tx: Prisma, team_id: str, user_id: str) -> Optional[dict[str, JsonValue]]:
        rows = await tx.query_raw(
            'SELECT to_jsonb(m) AS data FROM "LiteLLM_TeamMembership" m WHERE team_id=$1 AND user_id=$2 FOR UPDATE',
            team_id,
            user_id,
        )
        return _JSON_OBJECT.validate_python(rows[0]["data"]) if rows else None

    async def refresh_team(self, team_id: str, *, include_holds: bool = False) -> dict[str, JsonValue]:
        async with self.client.db.tx() as tx:
            team = await self.team_state(tx, team_id)
            self.validate_team(team)
            if include_holds and team.get("max_budget") is not None and cast(float, team["max_budget"]) > 0:
                return {
                    **team,
                    "spend": float(cast(float, team["spend"])) + await self.outstanding(tx, "team", team_id),
                }
            return team

    async def reset_teams(self) -> None:
        rows = await self.client.db.query_raw(
            'SELECT team_id FROM "LiteLLM_TeamTable" WHERE budget_duration IS NOT NULL '
            "AND (budget_reset_at IS NULL OR budget_reset_at <= clock_timestamp()) ORDER BY team_id"
        )
        for row in rows:
            await self.refresh_team(row["team_id"])

    @staticmethod
    def validate_team(team: dict[str, JsonValue]) -> None:
        team = _native_policy_data(team, models.LiteLLM_TeamTable)
        if any(
            team.get(field) is not None
            for field in (
                "tpm_limit",
                "rpm_limit",
                "max_parallel_requests",
                "soft_budget",
                "team_member_budget",
                "team_member_rpm_limit",
                "team_member_tpm_limit",
                "team_member_budget_duration",
                "model_aliases",
                "object_permission",
            )
        ) or any(
            team.get(field)
            for field in (
                "organization_id",
                "budget_id",
                "budget_limits",
                "model_max_budget",
                "model_id",
                "model_aliases",
                "model_rpm_limit",
                "model_tpm_limit",
                "object_permission",
                "object_permission_id",
                "default_team_member_models",
                "team_member_permissions",
                "team_member_key_duration",
                "access_group_ids",
                "policies",
                "tags",
                "guardrails",
                "prompts",
                "mcp_rpm_limit",
                "allowed_passthrough_routes",
                "disable_global_guardrails",
                "secret_manager_settings",
                "allowed_vector_store_indexes",
                "enforced_batch_output_expires_after",
                "enforced_file_expires_after",
                "allow_team_guardrail_config",
            )
        ):
            raise HTTPException(503, "PostgreSQL accounting: Team member/window/rate/organization policy pending")
        for field in ("metadata", "router_settings"):
            value = team.get(field) or {}
            data = _JSON_OBJECT.validate_json(value) if isinstance(value, str) else _JSON_OBJECT.validate_python(value)
            if (field == "router_settings" and any(data.values())) or (
                field == "metadata"
                and any(
                    data.get(name) is not None
                    for name in (
                        "team_member_budget_id",
                        "budget_id",
                        "tags",
                        "guardrails",
                        "model_max_budget",
                        "model_rpm_limit",
                        "model_tpm_limit",
                        "team_member_key_duration",
                        "default_key_params",
                        "policies",
                        "prompts",
                        "allowed_routes",
                        "passthrough_routes",
                    )
                )
            ):
                raise HTTPException(503, "PostgreSQL accounting: Team metadata policy pending")
        for field in ("spend", "max_budget"):
            value = team.get(field)
            if value is not None and not math.isfinite(float(cast(float, value))):
                raise HTTPException(400, "Non-finite native Team financial state")
        if team.get("budget_duration") is not None:
            PostgresAccounting.next_reset(str(team["budget_duration"]), datetime(2026, 1, 1, tzinfo=timezone.utc))

    @staticmethod
    def validate_member(member: Optional[dict[str, JsonValue]]) -> None:
        if member is not None and member.get("budget_id") is not None:
            raise HTTPException(503, "PostgreSQL accounting: linked Team member policy pending")

    async def outstanding(self, tx: Prisma, kind: Literal["key", "user", "team"], entity_id: str) -> float:
        rows = await tx.query_raw(
            'SELECT COALESCE(SUM(h.amount), 0)::float8 AS amount FROM "LiteLLM_AccountingHold" h '
            'JOIN "LiteLLM_AccountingScope" s ON s.id=h.scope_id WHERE s.kind=$1 AND s.entity_id=$2',
            kind,
            entity_id,
        )
        return float(rows[0]["amount"])

    async def refresh_user(self, user_id: str, *, include_holds: bool = False) -> Optional[dict[str, JsonValue]]:
        async with self.client.db.tx() as tx:
            user = await self.user_state(tx, user_id)
            if user is not None and include_holds:
                return {
                    **user,
                    "spend": float(cast(float, user["spend"])) + await self.outstanding(tx, "user", user_id),
                }
            return user

    async def reset_users(self) -> None:
        rows = await self.client.db.query_raw(
            'SELECT user_id FROM "LiteLLM_UserTable" WHERE budget_duration IS NOT NULL '
            "AND (budget_reset_at IS NULL OR budget_reset_at <= clock_timestamp()) ORDER BY user_id"
        )
        for row in rows:
            await self.refresh_user(row["user_id"])

    async def write_user(self, user_id: str, create: object, update: object) -> BaseModel:
        data = {"user_id": user_id, **_JSON_OBJECT.validate_json(safe_dumps(create))}
        changes = _JSON_OBJECT.validate_json(safe_dumps(update))
        if data["user_id"] != user_id or changes.get("user_id", user_id) != user_id:
            raise HTTPException(400, "Native user identity cannot be reassigned")
        async with self.client.db.tx() as tx:
            current = await self.user_state(tx, user_id)
            effective = changes if current is not None else data
            self.validate_user({**(current or {}), **effective})
            if "budget_duration" in effective:
                duration = effective["budget_duration"]
                boundary = self.next_reset(str(duration), await self.database_time(tx)) if duration else None
                effective = {**effective, "budget_reset_at": boundary.isoformat() if boundary else None}
            return await tx.litellm_usertable.upsert(
                where={"user_id": user_id},
                data={
                    "create": cast(
                        LiteLLM_UserTableCreateInput, self.client.jsonify_object(effective if current is None else data)
                    ),
                    "update": cast(
                        LiteLLM_UserTableUpdateInput,
                        self.client.jsonify_object(effective if current is not None else changes),
                    ),
                },
            )

    @staticmethod
    def validate_user(user: dict[str, JsonValue]) -> None:
        user = _native_policy_data(user, models.LiteLLM_UserTable)
        if any(
            user.get(field) is not None
            for field in ("tpm_limit", "rpm_limit", "max_parallel_requests", "model_rpm_limit", "model_tpm_limit")
        ) or any(
            user.get(field)
            for field in (
                "model_max_budget",
                "budget_limits",
                "organization_id",
                "organization_memberships",
                "organizations",
                "team_id",
            )
        ):
            raise HTTPException(503, "PostgreSQL accounting: native user rate/window/organization limits pending")
        for field in ("spend", "max_budget"):
            value = user.get(field)
            if value is not None and not math.isfinite(float(cast(float, value))):
                raise HTTPException(400, "Non-finite native user financial state")
        if user.get("budget_duration") is not None:
            PostgresAccounting.next_reset(str(user["budget_duration"]), datetime(2026, 1, 1, tzinfo=timezone.utc))

    async def lock_request_scopes(self, tx: Prisma, row: dict[str, JsonValue]) -> None:
        if row["key_token"]:
            await self.key_state(tx, cast(str, row["key_token"]))
        if row["user_id"]:
            await self.user_state(tx, cast(str, row["user_id"]))
        if row["team_id"]:
            await self.team_state(tx, cast(str, row["team_id"]))
            if row["user_id"]:
                await self.member_state(tx, cast(str, row["team_id"]), cast(str, row["user_id"]))

    async def replace_liability(
        self, tx: Prisma, request_id: str, component_id: str, amount: float, disposition: str
    ) -> None:
        components = await tx.query_raw(
            'SELECT id, liability FROM "LiteLLM_AccountingComponent" WHERE request_id=$1 ORDER BY id', request_id
        )
        previous = math.fsum(c["liability"] for c in components)
        holds = await tx.query_raw('SELECT amount FROM "LiteLLM_AccountingHold" WHERE request_id=$1', request_id)
        if (not holds and previous != 0) or any(
            not math.isclose(h["amount"], previous, rel_tol=1e-12, abs_tol=1e-12) for h in holds
        ):
            raise ValueError("Accounting membership liability mismatch")
        remaining = math.fsum(c["liability"] for c in components if c["id"] != component_id) + amount
        await tx.execute_raw(
            'UPDATE "LiteLLM_AccountingHold" SET amount=$2, disposition=$3 WHERE request_id=$1',
            request_id,
            remaining,
            disposition,
        )
        await tx.execute_raw('UPDATE "LiteLLM_AccountingComponent" SET liability=$2 WHERE id=$1', component_id, amount)

    async def refresh_key(self, token: str) -> dict[str, JsonValue]:
        async with self.client.db.tx() as tx:
            return await self.key_state(tx, token)

    async def reset_keys(self) -> None:
        rows = await self.client.db.query_raw(
            'SELECT token FROM "LiteLLM_VerificationToken" WHERE budget_duration IS NOT NULL '
            "AND (budget_reset_at IS NULL OR budget_reset_at <= clock_timestamp()) ORDER BY token"
        )
        for row in rows:
            await self.refresh_key(row["token"])

    @staticmethod
    def management_actor_token(auth: UserAPIKeyAuth) -> Optional[str]:
        from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS

        if auth.api_key == LITELLM_PROXY_MASTER_KEY_ALIAS and auth.user_role == "proxy_admin":
            return None
        if not auth.token:
            raise HTTPException(403, "Accounting management requires a native actor key")
        return auth.token

    async def management_keys(
        self, tx: Prisma, auth: UserAPIKeyAuth, *tokens: str, creating: Optional[str] = None
    ) -> dict[str, dict[str, JsonValue]]:
        actor_token = self.management_actor_token(auth)
        keys = {}
        for token in sorted(
            set(tokens) | ({actor_token} if actor_token else set()) | ({creating} if creating else set())
        ):
            if token != creating:
                keys[token] = await self.key_state(tx, token)
                continue
            stable = f"key:{token}:initial"
            await tx.execute_raw(
                "INSERT INTO \"LiteLLM_AccountingScope\" (id,kind,entity_id,epoch) VALUES ($1,'key',$2,'initial') ON CONFLICT DO NOTHING",
                stable,
                token,
            )
            await tx.query_raw('SELECT id FROM "LiteLLM_AccountingScope" WHERE id=$1 FOR UPDATE', stable)
            rows = await tx.query_raw(
                'SELECT to_jsonb(k) AS data FROM "LiteLLM_VerificationToken" k WHERE token=$1 FOR UPDATE', token
            )
            if rows:
                keys[token] = _JSON_OBJECT.validate_python(rows[0]["data"])
                self.validate_key(keys[token])
        if actor_token:
            actor_key = keys.get(actor_token)
            if actor_key is None or actor_key.get("user_id") != auth.user_id or actor_key.get("blocked"):
                raise HTTPException(403, "Accounting actor key authority changed")
            expires = actor_key.get("expires")
            if expires and datetime.fromisoformat(str(expires)).replace(
                tzinfo=timezone.utc
            ) <= await self.database_time(tx):
                raise HTTPException(403, "Accounting actor key expired")
        return keys

    async def management_users(
        self, tx: Prisma, auth: UserAPIKeyAuth, *user_ids: Optional[str]
    ) -> tuple[UserAPIKeyAuth, dict[str, Optional[dict[str, JsonValue]]]]:
        users = {
            user_id: await self.user_state(tx, user_id)
            for user_id in sorted({u for u in (*user_ids, auth.user_id) if u})
        }
        if self.management_actor_token(auth) is None:
            return auth, users
        actor = users.get(auth.user_id)
        if actor is None:
            raise HTTPException(403, "Accounting actor User missing")
        return auth.model_copy(update={"user_role": actor.get("user_role")}), users

    async def create_team(self, data: object, members: object, auth: UserAPIKeyAuth) -> BaseModel:
        from litellm.proxy._types import LiteLLM_UserTable, Member, NewTeamRequest
        from litellm.proxy.management_endpoints.team_endpoints import _check_user_team_limits
        from litellm.proxy.management_helpers.utils import add_team_to_user, get_new_internal_user_defaults
        from litellm.proxy.proxy_server import user_api_key_cache

        values = _native_policy_data(_JSON_OBJECT.validate_json(safe_dumps(data)), models.LiteLLM_TeamTable)
        self.validate_team(values)
        member_list = TypeAdapter(list[Member]).validate_python(members)
        if any(not m.user_id for m in member_list):
            raise HTTPException(503, "PostgreSQL accounting: email-only Team membership pending")
        team_id = cast(str, values["team_id"])
        async with self.client.db.tx(timeout=timedelta(seconds=60)) as tx:
            await self.management_keys(tx, auth)
            actor, users = await self.management_users(tx, auth, *(m.user_id for m in member_list))
            defaults = {
                m.user_id: _JSON_OBJECT.validate_python(get_new_internal_user_defaults(m.user_id)) for m in member_list
            }
            for user_id, default in defaults.items():
                self.validate_user(users[user_id] or default)
            if actor.user_role != "proxy_admin":
                await _check_user_team_limits(
                    NewTeamRequest(**{**values, "members_with_roles": member_list}),
                    actor,
                    self.client,
                    user_api_key_cache,
                    user_object=LiteLLM_UserTable(**users[actor.user_id]),
                )
            if values.get("budget_duration"):
                values = {
                    **values,
                    "budget_reset_at": self.next_reset(
                        str(values["budget_duration"]), await self.database_time(tx)
                    ).isoformat(),
                }
            row = await tx.litellm_teamtable.create(
                data=cast(
                    LiteLLM_TeamTableCreateInput,
                    self.client.jsonify_team_object(
                        {
                            **values,
                            "members_with_roles": [m.model_dump() for m in member_list],
                        }
                    ),
                )
            )
            for user_id in sorted(defaults):
                default = defaults[user_id]
                if default.get("budget_duration"):
                    default = {
                        **default,
                        "budget_reset_at": self.next_reset(
                            str(default["budget_duration"]), await self.database_time(tx)
                        ).isoformat(),
                    }
                await add_team_to_user(
                    tx, user_id, team_id, cast(LiteLLM_UserTableCreateInput, self.client.jsonify_object(default))
                )
            return row

    async def update_team(self, team_id: str, changes: object, auth: UserAPIKeyAuth) -> BaseModel:
        from litellm.proxy._types import LiteLLM_TeamTable, UpdateTeamRequest
        from litellm.proxy.management_endpoints.team_endpoints import (
            _check_team_budget_update_authority,
            _verify_team_access,
        )

        data = _native_policy_data(_JSON_OBJECT.validate_json(safe_dumps(changes)), models.LiteLLM_TeamTable)
        self.validate_team_update(data)
        async with self.client.db.tx() as tx:
            await self.management_keys(tx, auth)
            actor, _ = await self.management_users(tx, auth)
            team = await self.team_state(tx, team_id)
            self.validate_team({**team, **data})
            await _verify_team_access(LiteLLM_TeamTable(**team), actor)
            _check_team_budget_update_authority(
                UpdateTeamRequest(**{**data, "team_id": team_id}),
                actor,
                TypeAdapter(Optional[float]).validate_python(team.get("max_budget")),
            )
            if "budget_duration" in data:
                boundary = (
                    self.next_reset(str(data["budget_duration"]), await self.database_time(tx))
                    if data["budget_duration"]
                    else None
                )
                data = {**data, "budget_reset_at": boundary.isoformat() if boundary else None}
            row = await tx.litellm_teamtable.update(
                where={"team_id": team_id},
                data=cast(LiteLLM_TeamTableUpdateInput, self.client.jsonify_team_object(data)),
            )
            if row is None:
                raise HTTPException(404, "Accounting Team disappeared")
            return row

    @staticmethod
    def validate_team_update(data: dict[str, JsonValue]) -> None:
        if data.keys() - {
            "team_id",
            "max_budget",
            "budget_duration",
            "budget_reset_at",
            "team_alias",
            "metadata",
            "models",
        }:
            raise HTTPException(503, "PostgreSQL accounting: Team update fields pending")
        PostgresAccounting.validate_team(data)

    async def create_key(self, values: object, auth: UserAPIKeyAuth) -> BaseModel:
        from litellm.proxy._types import GenerateKeyRequest, KeyManagementRoutes, LiteLLM_TeamTableCachedObj
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            _team_key_generation_check,
            _check_team_key_limits,
            _personal_key_generation_check,
        )
        from litellm.proxy.management_endpoints.team_endpoints import _verify_team_access

        prepared = self.client.prepare_key_insert(TypeAdapter(dict[str, object]).validate_python(values))
        data = _native_policy_data(_JSON_OBJECT.validate_json(safe_dumps(prepared)), models.LiteLLM_VerificationToken)
        token = cast(str, data["token"])
        self.validate_key(data)
        async with self.client.db.tx() as tx:
            keys = await self.management_keys(tx, auth, creating=token)
            current = keys.get(token, {})
            actor, _ = await self.management_users(
                tx, auth, cast(Optional[str], data.get("user_id")), cast(Optional[str], current.get("user_id"))
            )
            actor_token = self.management_actor_token(auth)
            if actor.user_role != "proxy_admin" and actor_token:
                ceiling = keys[actor_token].get("max_budget")
                if (
                    ceiling is not None
                    and data.get("max_budget") is not None
                    and cast(float, data["max_budget"]) > cast(float, ceiling)
                ):
                    raise HTTPException(403, "Accounting key delegation ceiling changed")
            team_id = cast(Optional[str], data.get("team_id"))
            source_team = cast(Optional[str], current.get("team_id"))
            teams = {t: await self.team_state(tx, t) for t in sorted({t for t in (source_team, team_id) if t})}
            for team in teams.values():
                self.validate_team(team)
            if source_team:
                await _verify_team_access(LiteLLM_TeamTableCachedObj(**teams[source_team]), actor)
            elif current and actor.user_role != "proxy_admin" and current.get("user_id") != actor.user_id:
                raise HTTPException(403, "Accounting duplicate key owner changed")
            if team_id:
                team = teams[team_id]
                team_obj = LiteLLM_TeamTableCachedObj(**team)
                key_data = GenerateKeyRequest(**data)
                _team_key_generation_check(team_obj, actor, key_data, KeyManagementRoutes.KEY_GENERATE)
                await _check_team_key_limits(team_obj, key_data, self.client)
                if data.get("user_id"):
                    self.validate_member(await self.member_state(tx, team_id, cast(str, data["user_id"])))
            else:
                if actor.user_role != "proxy_admin" and data.get("user_id") != actor.user_id:
                    raise HTTPException(403, "Accounting personal key owner changed")
                _personal_key_generation_check(actor, GenerateKeyRequest(**data))
            if data.get("budget_duration"):
                prepared = {
                    **prepared,
                    "budget_reset_at": self.next_reset(str(data["budget_duration"]), await self.database_time(tx)),
                }
            return await self.client.insert_key(prepared, db=tx)

    async def update_key(
        self, token: str, changes: object, *, auth: Optional[UserAPIKeyAuth] = None
    ) -> dict[str, object]:
        from litellm.proxy.utils import hash_token

        token_hash = hash_token(token) if token.startswith("sk-") else token
        data = _JSON_OBJECT.validate_python(changes)
        allowed = {"max_budget", "budget_duration", "models", "metadata", "blocked", "key_alias", "user_id", "team_id"}
        if data.keys() - allowed:
            raise HTTPException(503, "PostgreSQL accounting: key update fields not qualified")
        async with self.client.db.tx() as tx:
            current = (
                (await self.management_keys(tx, auth, token_hash))[token_hash]
                if auth is not None
                else await self.key_state(tx, token_hash)
            )
            if auth is not None:
                from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_VerificationToken, UpdateKeyRequest
                from litellm.proxy.management_endpoints.key_management_endpoints import (
                    _validate_caller_can_change_key_ownership,
                    validate_key_team_change,
                )
                from litellm.proxy.management_endpoints.team_endpoints import _verify_team_access
                from litellm.proxy.proxy_server import llm_router

                actor, _ = await self.management_users(
                    tx, auth, cast(Optional[str], current.get("user_id")), cast(Optional[str], data.get("user_id"))
                )
                _validate_caller_can_change_key_ownership(
                    UpdateKeyRequest(key=token_hash, **data), LiteLLM_VerificationToken(**current), actor
                )
                team_ids = sorted({cast(str, t) for t in (current.get("team_id"), data.get("team_id")) if t})
                teams = {team_id: await self.team_state(tx, team_id) for team_id in team_ids}
                for team in teams.values():
                    self.validate_team(team)
                    await _verify_team_access(LiteLLM_TeamTable(**team), actor)
                if (
                    not current.get("team_id")
                    and actor.user_role != "proxy_admin"
                    and current.get("user_id") != actor.user_id
                ):
                    raise HTTPException(403, "Accounting key ownership changed; update no longer authorized")
                destination = cast(Optional[str], data.get("team_id", current.get("team_id")))
                if destination:
                    await validate_key_team_change(
                        LiteLLM_VerificationToken(**{**current, **data}),
                        LiteLLM_TeamTable(**teams[destination]),
                        actor,
                        llm_router,
                    )
                    user_id = cast(Optional[str], data.get("user_id", current.get("user_id")))
                    if user_id:
                        self.validate_member(await self.member_state(tx, destination, user_id))
            if isinstance(data.get("metadata"), dict):
                previous = current.get("metadata") or {}
                previous = json.loads(previous) if isinstance(previous, str) else previous
                data = {
                    **data,
                    "metadata": {
                        **_JSON_OBJECT.validate_python(previous),
                        **cast(dict[str, JsonValue], data["metadata"]),
                    },
                }
            self.validate_key({**current, **data})
            if "budget_duration" in data:
                duration = data["budget_duration"]
                now = await self.database_time(tx)
                reset_at = self.next_reset(str(duration), now) if duration else None
                await tx.execute_raw(
                    'UPDATE "LiteLLM_VerificationToken" SET budget_reset_at=$2::timestamp WHERE token=$1',
                    token_hash,
                    reset_at.isoformat() if reset_at else None,
                )
            row = await tx.litellm_verificationtoken.update(
                where={"token": token_hash},
                data=cast(LiteLLM_VerificationTokenUpdateInput, self.client.jsonify_object(data)),
            )
            if row is None:
                raise HTTPException(503, "Accounting key disappeared")
            return {"data": row.model_dump()}

    async def admit(
        self,
        token: Optional[str],
        estimate: Optional[float],
        input_cost: Optional[float],
        *,
        model: Optional[str] = None,
        auth: Optional[UserAPIKeyAuth] = None,
    ) -> None:
        request = self.context()
        if (
            not token
            or estimate is None
            or not math.isfinite(estimate)
            or estimate < 0
            or (input_cost is not None and (not math.isfinite(input_cost) or input_cost < 0))
        ):
            raise HTTPException(503, "PostgreSQL accounting: unqualified identity or pricing")
        async with self.client.db.tx() as tx:
            await tx.query_raw('SELECT id FROM "LiteLLM_AccountingRequest" WHERE id=$1 FOR UPDATE', request.request_id)
            key = await self.key_state(tx, token)
            if key.get("blocked") or (
                model is not None and key.get("models") and model not in cast(list, key["models"])
            ):
                raise HTTPException(403, "Accounting key blocked or model not allowed")
            expires = key.get("expires")
            if expires and datetime.fromisoformat(str(expires)).replace(
                tzinfo=timezone.utc
            ) <= await self.database_time(tx):
                raise HTTPException(403, "Accounting key expired")
            user_id = cast(Optional[str], key.get("user_id"))
            team_id = cast(Optional[str], key.get("team_id"))
            request.token, request.user_id, request.team_id = token, user_id, team_id
            user = await self.user_state(tx, user_id) if user_id else None
            team = await self.team_state(tx, team_id) if team_id else None
            if team is not None:
                from litellm.proxy._types import LiteLLM_TeamTable
                from litellm.proxy.auth.auth_checks import can_team_access_model, team_budget_exceeded
                from litellm.proxy.proxy_server import llm_router

                self.validate_team(team)
                if user_id:
                    self.validate_member(await self.member_state(tx, team_id, user_id))
                if team.get("blocked"):
                    raise HTTPException(403, "Accounting Team blocked")
                if model is not None:
                    await can_team_access_model(model, LiteLLM_TeamTable(**team), llm_router)
                cap = TypeAdapter(Optional[float]).validate_python(team.get("max_budget"))
                booked = float(cast(float, team["spend"]))
                if team_budget_exceeded(booked, cap):
                    raise litellm.BudgetExceededError(current_cost=booked, max_budget=cap)
            if user is not None:
                self.validate_user(user)
                if model is not None and team is None:
                    from litellm.proxy.auth.auth_checks import can_user_call_model
                    from litellm.proxy.proxy_server import llm_router
                    from litellm.proxy._types import LiteLLM_UserTable

                    await can_user_call_model(model=model, llm_router=llm_router, user_object=LiteLLM_UserTable(**user))
            reset_at = key.get("budget_reset_at")
            epoch = (
                datetime.fromisoformat(str(reset_at)).replace(tzinfo=timezone.utc).isoformat(timespec="milliseconds")
                if reset_at
                else "initial"
            )
            scope_id = f"key:{token}:{epoch}"
            await tx.execute_raw(
                "INSERT INTO \"LiteLLM_AccountingScope\" (id, kind, entity_id, epoch) VALUES ($1, 'key', $2, $3) ON CONFLICT DO NOTHING",
                scope_id,
                token,
                epoch,
            )
            from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, UserAPIKeyAuth
            from litellm.proxy.proxy_server import user_api_key_cache
            from litellm.proxy.spend_tracking.budget_reservation import _get_budget_counters

            counters = await _get_budget_counters(
                request_body={},
                valid_token=UserAPIKeyAuth(**key),
                team_object=LiteLLM_TeamTable(**team) if team is not None else None,
                user_object=LiteLLM_UserTable(**user) if user is not None else None,
                prisma_client=self.client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=self.client.proxy_logging_obj,
            )
            members = {
                ("Key", token): key,
                **(
                    {("Team", team_id): team}
                    if team is not None
                    else {("User", user_id): user}
                    if user is not None
                    else {}
                ),
            }
            reserved = estimate
            for counter in counters:
                if (counter.entity_type, counter.entity_id) not in members or counter.window_start is not None:
                    raise HTTPException(503, "PostgreSQL accounting: scope adapter not qualified")
                booked = counter.fallback_spend
                held = await self.outstanding(
                    tx, {"Key": "key", "User": "user", "Team": "team"}[counter.entity_type], counter.entity_id
                )
                remaining = counter.max_budget - booked - held
                if remaining <= 1e-12:
                    raise litellm.BudgetExceededError(current_cost=booked + held, max_budget=counter.max_budget)
                reserved = min(reserved, remaining)
            for (kind, _), member in members.items():
                budget = TypeAdapter(Optional[float]).validate_python(member.get("max_budget"))
                booked = float(cast(float, member["spend"]))
                if not math.isfinite(booked) or (budget is not None and not math.isfinite(budget)):
                    raise HTTPException(503, "PostgreSQL accounting: non-finite budget state")
                if kind != "Team" and budget is not None and budget <= 0:
                    raise litellm.BudgetExceededError(current_cost=booked, max_budget=budget)
            await tx.execute_raw(
                'UPDATE "LiteLLM_AccountingRequest" SET key_token=$2, input_cost=$3, user_id=$4, team_id=$5 WHERE id=$1',
                request.request_id,
                token,
                min(input_cost or 0.0, reserved),
                user_id,
                team_id,
            )
            memberships = (
                (scope_id, f"team:{team_id}:initial")
                if team_id
                else (scope_id, f"user:{user_id}:initial")
                if user_id
                else (scope_id,)
            )
            for membership in memberships:
                await tx.execute_raw(
                    'INSERT INTO "LiteLLM_AccountingHold" (request_id, scope_id, amount) VALUES ($1, $2, $3)',
                    request.request_id,
                    membership,
                    reserved,
                )
            await tx.execute_raw(
                'UPDATE "LiteLLM_AccountingComponent" SET liability=$2 WHERE id=$1', request.component_id, reserved
            )
        request.reserved_cost = reserved
        request.token = token
        request.user_id = cast(Optional[str], key.get("user_id"))
        metadata = key.get("metadata") or {}
        request.key_metadata = _JSON_OBJECT.validate_python(
            json.loads(metadata) if isinstance(metadata, str) else metadata
        )
        if auth is not None:
            auth.metadata = dict(request.key_metadata)
            auth.user_id = request.user_id
            auth.team_id = request.team_id
            auth.team_metadata = LiteLLM_TeamTable(**team).metadata if team is not None else {}
            auth.team_alias = cast(Optional[str], team.get("team_alias")) if team is not None else None
            auth.team_spend = float(cast(float, team["spend"])) if team is not None else None
            auth.team_max_budget = (
                TypeAdapter(Optional[float]).validate_python(team.get("max_budget")) if team is not None else None
            )
        request.input_cost = min(input_cost or 0.0, reserved)
        request.admitted = True

    @staticmethod
    def validate_key(key: dict[str, JsonValue]) -> None:
        key = _native_policy_data(key, models.LiteLLM_VerificationToken)
        unsupported = (
            "organization_id",
            "project_id",
            "agent_id",
            "budget_id",
            "budget_limits",
            "tpm_limit",
            "rpm_limit",
            "max_parallel_requests",
            "model_max_budget",
            "config",
        )
        if any(key.get(field) for field in unsupported) or any(
            key.get(field) is not None for field in ("tpm_limit", "rpm_limit", "max_parallel_requests")
        ):
            raise HTTPException(
                503, "PostgreSQL accounting first vertical: parent, reset/window, model and limiter integration pending"
            )
        router = key.get("router_settings")
        if isinstance(router, dict) and any(router.values()):
            raise HTTPException(503, "PostgreSQL accounting first vertical: per-key routing integration pending")
        metadata = key.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if metadata:
            trusted_metadata = _JSON_OBJECT.validate_python(metadata)
            if any(
                trusted_metadata.get(name) for name in ("tags", "budget_id", "guardrails", "model_group", "model_info")
            ):
                raise HTTPException(503, "PostgreSQL accounting: key metadata scope/routing integration pending")
        if key.get("budget_duration") is not None:
            PostgresAccounting.next_reset(str(key["budget_duration"]), datetime(2026, 1, 1, tzinfo=timezone.utc))

    async def dispatch(self) -> None:
        request = self.context()
        if not request.admitted or request.failed:
            raise HTTPException(503, "Accounting admission or producer state unavailable")
        component_id = str(uuid4()) if request.dispatched else request.component_id
        additional = request.dispatched
        request.dispatched = True
        request.last_component_id = component_id
        call = accounting_call.get()
        if call is not None:
            call.component_id = component_id
        try:
            async with self.client.db.tx() as tx:
                rows = await tx.query_raw(
                    'SELECT * FROM "LiteLLM_AccountingRequest" WHERE id=$1 FOR UPDATE', request.request_id
                )
                await self.lock_request_scopes(tx, rows[0])
                if additional:
                    await tx.execute_raw(
                        'INSERT INTO "LiteLLM_AccountingComponent" (id, request_id, dispatched) VALUES ($1,$2,true)',
                        component_id,
                        request.request_id,
                    )
                    await self.replace_liability(tx, request.request_id, component_id, request.reserved_cost, "held")
                else:
                    await tx.execute_raw(
                        'UPDATE "LiteLLM_AccountingComponent" SET dispatched=true WHERE id=$1', component_id
                    )
                await tx.execute_raw(
                    'UPDATE "LiteLLM_AccountingRequest" SET dispatched=true WHERE id=$1', request.request_id
                )
        except BaseException:
            request.failed = True
            raise

    async def reject_component(self, component_id: str) -> None:
        request = self.context()
        async with self.client.db.tx() as tx:
            rows = await tx.query_raw(
                'SELECT * FROM "LiteLLM_AccountingRequest" WHERE id=$1 FOR UPDATE', request.request_id
            )
            await self.lock_request_scopes(tx, rows[0])
            components = await tx.query_raw(
                'SELECT * FROM "LiteLLM_AccountingComponent" WHERE id=$1 AND request_id=$2',
                component_id,
                request.request_id,
            )
            if not components or components[0]["actual"] or not components[0]["dispatched"]:
                raise ValueError("Invalid rejected accounting component")
            await self.replace_liability(tx, request.request_id, component_id, 0.0, "not_executed")
            await tx.execute_raw(
                'UPDATE "LiteLLM_AccountingComponent" SET liability=0, nonexecution=true WHERE id=$1', component_id
            )

    def component_id(self) -> str:
        request = self.context()
        call = accounting_call.get()
        return (call.component_id if call else None) or request.last_component_id or request.component_id

    async def routing_usage(self, model_group: str) -> tuple[dict[str, int], dict[str, int]]:
        rows = await self.client.db.query_raw(
            'SELECT l.model_id, SUM(l.total_tokens)::int AS tpm, COUNT(*)::int AS rpm FROM "LiteLLM_SpendLogs" l '
            "JOIN \"LiteLLM_AccountingReceipt\" r ON r.component_id=l.request_id AND r.basis='actual' "
            "WHERE l.model_group=$1 AND r.created_at >= date_trunc('minute', clock_timestamp() AT TIME ZONE 'UTC') GROUP BY l.model_id",
            model_group,
        )
        return ({r["model_id"]: r["tpm"] for r in rows}, {r["model_id"]: r["rpm"] for r in rows})

    async def current_spend(self, counter_key: str) -> float:
        if not counter_key.startswith("spend:key:") or ":window:" in counter_key:
            raise HTTPException(503, "PostgreSQL accounting first vertical: scope integration pending")
        token = counter_key.removeprefix("spend:key:")
        await self.refresh_key(token)
        rows = await self.client.db.query_raw(
            'SELECT spend + (SELECT COALESCE(SUM(h.amount), 0) FROM "LiteLLM_AccountingHold" h JOIN "LiteLLM_AccountingScope" s ON s.id=h.scope_id WHERE s.kind=\'key\' AND s.entity_id=k.token) AS spend FROM "LiteLLM_VerificationToken" k WHERE token=$1',
            token,
        )
        if not rows:
            raise HTTPException(503, "PostgreSQL accounting: authoritative key unavailable")
        return float(rows[0]["spend"])

    def context(self) -> AccountingRequest:
        request = accounting_request.get()
        if request is None or request.store is not self:
            raise RuntimeError("PostgreSQL accounting: missing server-owned request context")
        return request

    async def enqueue(
        self, raw: object, daily: object, cost: Optional[float], *, basis: Literal["actual", "estimate"] = "actual"
    ) -> None:
        request = self.context()
        if request.producers == 0:
            raise ValueError("Accounting observation after producer sealing")
        if cost is None or not math.isfinite(cost) or cost < 0:
            request.failed = True
            raise ValueError("Accounting observation requires a finite nonnegative cost")
        raw_data = _JSON_OBJECT.validate_json(safe_dumps(raw))
        daily_data = _JSON_OBJECT.validate_json(safe_dumps(daily))
        metadata = raw_data.get("metadata") or {}
        raw_metadata = (
            _JSON_OBJECT.validate_json(metadata)
            if isinstance(metadata, str)
            else _JSON_OBJECT.validate_python(metadata)
        )
        component_id = self.component_id()
        envelope = Envelope(
            request_id=request.request_id,
            component_id=component_id,
            receipt_id=str(uuid5(UUID(component_id), basis)),
            basis=basis,
            total=cost,
            raw={
                **raw_data,
                "request_id": component_id,
                "api_key": request.token or "",
                "user": request.user_id or "",
                "team_id": request.team_id,
                "metadata": {
                    **raw_metadata,
                    "reported_request_id": raw_data.get("request_id"),
                    "accounting_request_id": request.request_id,
                    "user_api_key_auth_metadata": request.key_metadata,
                },
            }
            if raw_data
            else {},
            daily={
                **daily_data,
                "api_key": request.token or "",
                "user_id": request.user_id or "",
                "team_id": request.team_id,
            }
            if daily_data
            else {},
        )
        total_identity = json.dumps(
            {
                "total": cost,
                "raw": {
                    key: envelope.raw.get(key)
                    for key in ("total_tokens", "prompt_tokens", "completion_tokens", "model", "cache_hit", "status")
                },
                "daily": envelope.daily,
            },
            sort_keys=True,
            allow_nan=False,
        )
        observation_id = f"{component_id}:{basis}"
        previous = request.observations.get(observation_id)
        if previous is not None:
            if previous != total_identity:
                request.failed = True
                raise ValueError("Conflicting component-total observation")
            return
        request.observations[observation_id] = total_identity
        request.expected[envelope.receipt_id] = envelope.fingerprint()
        spool = self.client._spend_log_spool
        if spool is None:
            request.failed = True
            raise RuntimeError("PostgreSQL accounting requires the durable SQLite outbox")
        from litellm.proxy.db.spend_log_queue import enqueue_spend_log

        await enqueue_spend_log(self.client, {"postgres_accounting": envelope.model_dump()})

    async def bind_failure_key(self, token: Optional[str], cost: Optional[float]) -> None:
        request = self.context()
        if request.admitted:
            return
        if cost != 0 or request.dispatched:
            raise ValueError("Unadmitted request cannot record provider cost")
        async with self.client.db.tx() as tx:
            await tx.query_raw('SELECT id FROM "LiteLLM_AccountingRequest" WHERE id=$1 FOR UPDATE', request.request_id)
            if request.token is None and token:
                key = await self.key_state(tx, token)
                request.token = token
                request.user_id = cast(Optional[str], key.get("user_id"))
                request.team_id = cast(Optional[str], key.get("team_id"))
            if request.token:
                await self.lock_request_scopes(
                    tx, {"key_token": request.token, "user_id": request.user_id, "team_id": request.team_id}
                )
                await tx.execute_raw(
                    'UPDATE "LiteLLM_AccountingRequest" SET key_token=$2, user_id=$3, team_id=$4 WHERE id=$1',
                    request.request_id,
                    request.token,
                    request.user_id,
                    request.team_id,
                )

    async def cancel(self) -> None:
        request = self.context()
        if request.admitted and request.dispatched:
            await self.enqueue({}, {}, request.input_cost, basis="estimate")

    async def seal(self, request: AccountingRequest) -> None:
        marker = ProducerSeal(
            request_id=request.request_id,
            component_id=request.component_id,
            receipt_id=str(uuid5(UUID(request.component_id), "seal")),
            expected=tuple(sorted(request.expected.items())),
            failed=request.failed,
            nonexecution=not request.dispatched,
        )
        spool = self.client._spend_log_spool
        if spool is None:
            raise RuntimeError("PostgreSQL accounting requires the durable SQLite outbox")
        await spool.enqueue({"postgres_accounting": marker.model_dump()})

    async def _apply_seal(self, marker: ProducerSeal) -> None:
        async with self._settlement_transaction() as tx:
            rows = await tx.query_raw(
                'SELECT * FROM "LiteLLM_AccountingRequest" WHERE id=$1 FOR UPDATE', marker.request_id
            )
            components = await tx.query_raw(
                'SELECT * FROM "LiteLLM_AccountingComponent" WHERE request_id=$1', marker.request_id
            )
            if (
                not rows
                or marker.component_id not in {c["id"] for c in components}
                or marker.receipt_id != str(uuid5(UUID(marker.component_id), "seal"))
            ):
                raise ValueError("Unknown producer seal request, component or receipt")
            receipts = await tx.query_raw(
                'SELECT receipt.id, receipt.basis, receipt.payload_hash, receipt.total FROM "LiteLLM_AccountingReceipt" receipt '
                'JOIN "LiteLLM_AccountingComponent" c ON c.id=receipt.component_id WHERE c.request_id=$1',
                marker.request_id,
            )
            existing = next((r for r in receipts if r["basis"] == "seal"), None)
            if existing is not None:
                if existing["payload_hash"] != marker.fingerprint():
                    raise ValueError("Producer seal payload conflict")
                return
            expected = dict(marker.expected)
            valid_ids = {str(uuid5(UUID(c["id"]), basis)) for c in components for basis in ("actual", "estimate")}
            if (
                tuple(sorted(expected.items())) != marker.expected
                or not expected.keys() <= valid_ids
                or any(expected.get(r["id"]) != r["payload_hash"] for r in receipts)
                or (
                    marker.nonexecution
                    and (
                        rows[0]["dispatched"]
                        or any(c["dispatched"] for c in components)
                        or any(r["total"] != 0 for r in receipts)
                    )
                )
            ):
                raise ValueError("Producer seal manifest conflict")
            if expected.keys() != {r["id"] for r in receipts}:
                raise AccountingReceiptsPending("Producer seal awaiting accounting receipts")
            if marker.nonexecution:
                await self.lock_request_scopes(tx, rows[0])
                for component in components:
                    await self.replace_liability(tx, marker.request_id, component["id"], 0.0, "not_executed")
                await tx.execute_raw(
                    'UPDATE "LiteLLM_AccountingComponent" SET liability=0, nonexecution=true WHERE request_id=$1',
                    marker.request_id,
                )
            await tx.execute_raw(
                "INSERT INTO \"LiteLLM_AccountingReceipt\" (id, component_id, basis, payload_hash, total) VALUES ($1,$2,'seal',$3,0)",
                marker.receipt_id,
                marker.component_id,
                marker.fingerprint(),
            )
            await tx.execute_raw(
                'UPDATE "LiteLLM_AccountingRequest" SET sealed=true, failed=failed OR $2, expected=$3::text[] WHERE id=$1',
                marker.request_id,
                marker.failed
                or (not marker.nonexecution and any(not c["actual"] and not c["nonexecution"] for c in components)),
                list(expected),
            )
            await self._close(tx, marker.request_id)

    @staticmethod
    async def _close(tx: Prisma, request_id: str) -> None:
        await tx.execute_raw(
            'UPDATE "LiteLLM_AccountingRequest" r SET closed=true WHERE r.id=$1 AND sealed AND NOT failed AND NOT EXISTS (SELECT 1 FROM unnest(r.expected) e(id) WHERE NOT EXISTS (SELECT 1 FROM "LiteLLM_AccountingReceipt" s WHERE s.id=e.id))',
            request_id,
        )

    async def apply(self, envelope: Envelope | ProducerSeal) -> None:
        try:
            if isinstance(envelope, ProducerSeal):
                await self._apply_seal(envelope)
            else:
                await self._apply(envelope)
        except ValueError:
            await self.client.db.execute_raw(
                'UPDATE "LiteLLM_AccountingRequest" SET failed=true, closed=false WHERE id=$1', envelope.request_id
            )
            raise

    async def _apply(self, envelope: Envelope) -> None:
        if not math.isfinite(envelope.total) or envelope.total < 0:
            raise ValueError("Invalid accounting total")
        async with self._settlement_transaction() as tx:
            requests = await tx.query_raw(
                'SELECT * FROM "LiteLLM_AccountingRequest" WHERE id=$1 FOR UPDATE', envelope.request_id
            )
            if not requests:
                raise ValueError("Unknown accounting request")
            components = await tx.query_raw(
                'SELECT * FROM "LiteLLM_AccountingComponent" WHERE id=$1 AND request_id=$2',
                envelope.component_id,
                envelope.request_id,
            )
            if not components or envelope.receipt_id != str(uuid5(UUID(envelope.component_id), envelope.basis)):
                raise ValueError("Unknown accounting component or receipt")
            if requests[0]["sealed"] and envelope.receipt_id not in requests[0]["expected"]:
                raise ValueError("Observation outside sealed producer manifest")
            receipts = await tx.query_raw(
                'SELECT payload_hash FROM "LiteLLM_AccountingReceipt" WHERE id=$1', envelope.receipt_id
            )
            if receipts:
                if receipts[0]["payload_hash"] != envelope.fingerprint():
                    raise ValueError("Accounting receipt payload conflict")
                return
            await self.lock_request_scopes(tx, requests[0])
            actual_already = components[0]["actual"]
            token = requests[0]["key_token"]
            if envelope.basis == "actual":
                if not envelope.raw or not envelope.daily:
                    raise ValueError("Actual settlement requires raw and daily mappings")
                if actual_already:
                    raise ValueError("Conflicting component total")
                user_id = requests[0]["user_id"] or ""
                team_id = requests[0]["team_id"]
                raw = {
                    **envelope.raw,
                    "api_key": token or "",
                    "user": user_id,
                    "team_id": team_id,
                    "spend": envelope.total,
                }
                await tx.litellm_spendlogs.create(
                    data=cast(LiteLLM_SpendLogsCreateInput, self.client.jsonify_object(raw))
                )
                from litellm.proxy._types import DailyTeamSpendTransaction, DailyUserSpendTransaction
                from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

                async with tx.batch_() as batcher:
                    if token:
                        DBSpendUpdateWriter.write_key_spend(batcher, token, envelope.total)
                    if user_id:
                        DBSpendUpdateWriter.write_user_spend(batcher, user_id, envelope.total)
                    if team_id:
                        DBSpendUpdateWriter.write_team_spend(batcher, team_id, envelope.total)
                        if user_id:
                            DBSpendUpdateWriter.write_team_member_spend(batcher, team_id, user_id, envelope.total)
                        DBSpendUpdateWriter.write_daily_spend(
                            batcher,
                            cast(
                                DailyTeamSpendTransaction,
                                {**envelope.daily, "api_key": token or "", "team_id": team_id},
                            ),
                            "team",
                            "team_id",
                            "litellm_dailyteamspend",
                            "team_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint",
                        )
                    DBSpendUpdateWriter.write_daily_spend(
                        batcher,
                        cast(DailyUserSpendTransaction, {**envelope.daily, "api_key": token or "", "user_id": user_id}),
                        "user",
                        "user_id",
                        "litellm_dailyuserspend",
                        "user_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint",
                    )
                await self.replace_liability(tx, envelope.request_id, envelope.component_id, 0.0, "settled")
                await tx.execute_raw(
                    'UPDATE "LiteLLM_AccountingComponent" SET actual=true, total=$2 WHERE id=$1',
                    envelope.component_id,
                    envelope.total,
                )
            else:
                if (
                    envelope.total != requests[0]["input_cost"]
                    or not requests[0]["dispatched"]
                    or envelope.raw
                    or envelope.daily
                ):
                    raise ValueError("Invalid cancellation estimate")
                if not actual_already and not components[0]["nonexecution"]:
                    await self.replace_liability(
                        tx, envelope.request_id, envelope.component_id, envelope.total, "estimated"
                    )
            await tx.execute_raw(
                'INSERT INTO "LiteLLM_AccountingReceipt" (id, component_id, basis, payload_hash, total) VALUES ($1,$2,$3,$4,$5)',
                envelope.receipt_id,
                envelope.component_id,
                envelope.basis,
                envelope.fingerprint(),
                envelope.total,
            )
            await self._close(tx, envelope.request_id)

    def unknown_status(self) -> AccountingStatus:
        return AccountingStatus(
            generation_id=self.generation_id,
            incarnation_id=self.incarnation_id,
            generation_bound=self.generation_bound,
            accepting=None,
            pending_requests=None,
            status="accounting_unknown",
        )

    async def status(self) -> AccountingStatus:
        rows = await self.client.db.query_raw(
            'SELECT g.version, g.accepting, (SELECT COUNT(*)::int FROM "LiteLLM_AccountingRequest" r '
            "WHERE r.generation_id=g.id AND NOT r.closed) AS pending "
            'FROM "LiteLLM_AccountingGeneration" g WHERE g.id=$1',
            self.generation_id,
        )
        if len(rows) != 1 or rows[0]["version"] != 4:
            raise RuntimeError("Missing or incompatible accounting generation")
        return AccountingStatus(
            generation_id=self.generation_id,
            incarnation_id=self.incarnation_id,
            generation_bound=self.generation_bound,
            accepting=rows[0]["accepting"],
            pending_requests=rows[0]["pending"],
            status="accounting_pending" if rows[0]["pending"] else "accounting_complete",
        )

    async def set_accepting(self, accepting: bool) -> AccountingStatus:
        changed = await self.client.db.execute_raw(
            'UPDATE "LiteLLM_AccountingGeneration" SET accepting=$2 WHERE id=$1 AND version=4',
            self.generation_id,
            accepting,
        )
        if changed != 1:
            raise RuntimeError("Missing or incompatible accounting generation")
        return await self.status()

    async def drain(self) -> int:
        result = await self.set_accepting(False)
        assert result.pending_requests is not None
        return result.pending_requests


runtime: Optional[PostgresAccounting] = None


async def initialize(client: Optional[PrismaClient], settings: object, router: object) -> None:
    global runtime
    config = _JSON_OBJECT.validate_python(settings)
    if not config.get("postgres_admission_accounting"):
        return
    if os.getenv("DATABASE_URL_READ_REPLICA"):
        raise RuntimeError(
            "PostgreSQL accounting first vertical requires writer-only reads; read replica integration pending"
        )
    if client is None or client._spend_log_spool is None:
        raise RuntimeError("PostgreSQL accounting requires PostgreSQL and SPEND_LOG_DURABLE_QUEUE_PATH")
    conflicts = (
        "disable_spend_logs",
        "disable_spend_updates",
        "disable_budget_reservation",
        "use_redis_transaction_buffer",
        "background_health_checks",
        "pass_through_endpoints",
    )
    if any(config.get(name) for name in conflicts) or os.getenv("SPEND_LOGS_URL"):
        raise RuntimeError(
            "PostgreSQL accounting conflicts with disabled or external accounting and dynamic configuration"
        )
    if config.get("disable_prisma_schema_update") is not True:
        raise RuntimeError("PostgreSQL accounting requires explicit migration")
    if litellm.max_budget or any(
        getattr(router, field, None)
        for field in (
            "fallbacks",
            "context_window_fallbacks",
            "content_policy_fallbacks",
            "model_group_retry_policy",
        )
    ):
        raise RuntimeError("PostgreSQL accounting first vertical: global budget and retry/fallback integration pending")
    from litellm.proxy.spend_tracking.postgres_accounting_config import validate_deployment

    for model in getattr(router, "model_list", ()):
        validate_deployment(model)
    if litellm.aclient_session is not None:
        raise RuntimeError("PostgreSQL accounting requires native instrumented provider clients")
    instance = PostgresAccounting(client, generation_id=os.getenv("LITELLM_ACCOUNTING_GENERATION_ID"))
    await instance.initialize()
    runtime = instance
