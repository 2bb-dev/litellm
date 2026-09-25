"""Discovery never advertises subscription routes to non-entitled virtual keys."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth

SUBSCRIPTION = (
    "anthropic/claude-opus-5-5/pi",
    "pi/claude-opus-5-5-chat",
    "claude-opus-5-5-pi-native",
    "claude-opus-5-5-pi-chat",
)
OTHER = ("anthropic/claude-opus-5-5", "gpt-4")
HASH = "a" * 64


@pytest.fixture
def catalog(monkeypatch):
    from litellm.proxy import utils

    monkeypatch.setenv("OPENORANGE_CLAUDE_SUBSCRIPTION_KEY_HASHES", HASH)
    router = MagicMock()
    router.get_fully_blocked_model_names.return_value = set()
    router.model_list = [
        {"model_name": name, "litellm_params": {"model": name}, "model_info": {"id": name}}
        for name in (*SUBSCRIPTION, *OTHER)
    ]
    router.get_model_list_from_model_alias.return_value = []
    router.get_model_names.return_value = list((*SUBSCRIPTION, *OTHER))
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", router.model_list)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "_enrich_model_info_with_litellm_data", lambda model, **kw: model)
    monkeypatch.setattr(utils, "get_available_models_for_user", AsyncMock(return_value=list((*SUBSCRIPTION, *OTHER))))
    monkeypatch.setattr(utils, "create_model_info_response", lambda model_id, **kw: {"id": model_id})
    return router


@pytest.mark.asyncio
@pytest.mark.parametrize("token,expected", [("b" * 64, OTHER), (HASH, (*SUBSCRIPTION, *OTHER)), (None, OTHER), ("invalid", OTHER)])
async def test_models_list_only_advertises_subscription_to_entitled_key(catalog, token, expected):
    response = await proxy_server.model_list(user_api_key_dict=UserAPIKeyAuth(token=token, models=[]))
    assert tuple(row["id"] for row in response["data"]) == expected


@pytest.mark.asyncio
async def test_missing_or_malformed_entitlement_hides_subscription(catalog, monkeypatch):
    for configured in ("", "not-a-hash", "A" * 64):
        monkeypatch.setenv("OPENORANGE_CLAUDE_SUBSCRIPTION_KEY_HASHES", configured)
        response = await proxy_server.model_list(user_api_key_dict=UserAPIKeyAuth(token=HASH, models=[]))
        assert tuple(row["id"] for row in response["data"]) == OTHER


@pytest.mark.asyncio
async def test_model_info_hides_subscription_deployments(catalog):
    response = await proxy_server.model_info_v1(user_api_key_dict=UserAPIKeyAuth(token="b" * 64, models=[]))
    assert tuple(row["model_name"] for row in response["data"]) == OTHER


@pytest.mark.asyncio
async def test_model_info_id_hides_subscription(catalog):
    catalog.get_deployment.return_value = MagicMock(model_dump=lambda **kw: catalog.model_list[0])
    with pytest.raises(HTTPException) as exc:
        await proxy_server.model_info_v1(user_api_key_dict=UserAPIKeyAuth(token="b" * 64, models=[]), litellm_model_id=SUBSCRIPTION[0])
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_model_group_info_hides_subscription(catalog):
    catalog.get_model_group_info.return_value = None
    response = await proxy_server.model_group_info(user_api_key_dict=UserAPIKeyAuth(token="b" * 64, models=[]))
    assert tuple(row.model_group for row in response["data"]) == OTHER

@pytest.mark.asyncio
@pytest.mark.parametrize("name", SUBSCRIPTION)
async def test_individual_model_lookup_hides_subscription(catalog, name):
    with pytest.raises(HTTPException):
        await proxy_server.model_info(model_id=name, user_api_key_dict=UserAPIKeyAuth(token="b" * 64, models=[]))


@pytest.mark.asyncio
async def test_admin_expanded_listing_does_not_override_entitlement(catalog, monkeypatch):
    monkeypatch.setattr(proxy_server, "_user_has_admin_view", lambda auth: True)
    catalog.get_model_access_groups.return_value = {}
    response = await proxy_server.model_list(user_api_key_dict=UserAPIKeyAuth(token="b" * 64, models=[]), scope="expand")
    assert tuple(row["id"] for row in response["data"]) == OTHER


@pytest.mark.asyncio
async def test_anthropic_format_hides_subscription(catalog):
    request = MagicMock()
    request.headers = {"anthropic-version": "2023-06-01"}
    response = await proxy_server.model_list(request=request, user_api_key_dict=UserAPIKeyAuth(token="b" * 64, models=[]))
    assert not any(name in str(response) for name in SUBSCRIPTION)

@pytest.mark.asyncio
async def test_v2_search_excludes_protected_db_rows_before_count_and_page(catalog, monkeypatch):
    repository = MagicMock()
    repository.table.count = AsyncMock(return_value=0)
    repository.table.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr(proxy_server, "ModelRepository", lambda client: repository)
    monkeypatch.setattr(proxy_server, "_get_caller_byok_team_scope", AsyncMock(return_value=None))
    await proxy_server._apply_search_filter_to_models(
        all_models=catalog.model_list, search="claude", prisma_client=MagicMock(),
        proxy_config=MagicMock(), user_api_key_dict=UserAPIKeyAuth(token="b" * 64),
        page=1, size=2, subscription_token="b" * 64,
    )
    where = repository.table.count.await_args.kwargs["where"]
    assert repository.table.find_many.await_args.kwargs["where"] == where
    assert "NOT" in where
    assert set(SUBSCRIPTION).issubset(set(str(where).replace("'", '"').split('"')[1::2]))


@pytest.mark.asyncio
async def test_v2_model_id_hidden_before_metadata_or_logging(catalog, monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server.proxy_config, "get_config", AsyncMock())
    finder = AsyncMock(return_value=([catalog.model_list[0]], 1))
    monkeypatch.setattr(proxy_server, "_find_model_by_id", finder)
    response = await proxy_server.model_info_v2(
        user_api_key_dict=UserAPIKeyAuth(token="b" * 64), modelId=SUBSCRIPTION[0], page=1, size=1,
    )
    assert response["total_count"] == 0
    assert response["data"] == []
    finder.assert_not_awaited()

@pytest.mark.asyncio
async def test_v2_search_preserves_filtered_db_count_across_pages(catalog, monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server.proxy_config, "get_config", AsyncMock())
    monkeypatch.setattr(proxy_server, "_apply_search_filter_to_models", AsyncMock(return_value=([catalog.model_list[-1]], 7)))
    response = await proxy_server.model_info_v2(
        user_api_key_dict=UserAPIKeyAuth(token="b" * 64), search="gpt", page=2, size=1, teamId=None, sortBy=None, modelId=None, model=None, user_models_only=False, include_team_models=False, access_group=None, wildcard_only=False, exclude_auto_routers=False,
    )
    assert response["total_count"] == 7
    assert response["total_pages"] == 7
    assert response["data"] == []

@pytest.mark.asyncio
async def test_team_public_alias_is_hidden_before_metadata_lookup(catalog, monkeypatch):
    internal = "model_name_team1_123"
    catalog.model_list.insert(0, {
        "model_name": internal,
        "litellm_params": {"model": "anthropic/claude-opus-5-5"},
        "model_info": {"id": "team-id", "team_id": "team1", "team_public_model_name": SUBSCRIPTION[0]},
    })
    catalog.get_model_list.return_value = catalog.model_list
    monkeypatch.setattr(proxy_server, "llm_model_list", catalog.model_list)
    from litellm.proxy import utils
    monkeypatch.setattr(utils, "get_available_models_for_user", AsyncMock(return_value=[internal, OTHER[0]]))
    response = await proxy_server.model_list(user_api_key_dict=UserAPIKeyAuth(token="b" * 64, models=[]))
    assert [row["id"] for row in response["data"]] == [OTHER[0]]
    assert not proxy_server.subscription_deployment_visible("b" * 64, catalog.model_list[0])
    with pytest.raises(HTTPException) as exc:
        await proxy_server.model_info(model_id=SUBSCRIPTION[0], user_api_key_dict=UserAPIKeyAuth(token="b" * 64, models=[]))
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_entitled_model_info_keeps_subscription(catalog):
    response = await proxy_server.model_info_v1(user_api_key_dict=UserAPIKeyAuth(token=HASH, models=[]))
    assert {row["model_name"] for row in response["data"]} == set((*SUBSCRIPTION, *OTHER))
