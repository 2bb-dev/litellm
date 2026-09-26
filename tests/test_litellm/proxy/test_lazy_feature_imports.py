"""Every lazily registered feature router must import and register.

LazyFeatureMiddleware logs a failed import, marks the feature as loaded and
then answers 404 for its routes until restart. A broken import, such as a
typing name lost in an upstream sync, otherwise only shows up as missing
endpoints on a running proxy.
"""

import importlib

import pytest
from fastapi import FastAPI

from litellm.proxy._lazy_features import LAZY_FEATURES, LazyFeature


@pytest.mark.parametrize("feature", LAZY_FEATURES, ids=lambda feature: feature.name)
def test_lazy_feature_imports_and_registers(feature: LazyFeature):
    app = FastAPI()
    before = len(app.router.routes)

    feature.register_fn(app, importlib.import_module(feature.module_path))

    assert len(app.router.routes) > before


def test_anthropic_messages_routes_register():
    feature = next(f for f in LAZY_FEATURES if f.name == "anthropic_passthrough")
    app = FastAPI()

    feature.register_fn(app, importlib.import_module(feature.module_path))

    paths = {getattr(route, "path", None) for route in app.router.routes}
    assert {"/v1/messages", "/v1/messages/count_tokens"} <= paths
