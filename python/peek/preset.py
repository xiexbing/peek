# Copyright 2026 Bing Xie
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PEEK preset env-var bundles.

Set ``PEEK_PRESET=peek-online`` (or ``peek-offline``) and the matching
``PEEK_*`` defaults are populated before the rest of peek reads the
environment. Explicit env vars already set by the user take precedence
(``os.environ.setdefault`` semantics), so per-flag ablations still work.

Bundles
-------
``peek-online``: paper's primary online configuration -- scheduler hook
on, Cluster-LPM, group-major, dynamic lane, and cluster-mode queue-aware
eviction.

``peek-offline``: full offline stack -- engine enable, server-side
reorder, and queue-aware eviction.
"""

import os
import warnings

_PRESETS: dict[str, dict[str, str]] = {
    "peek-online": {
        "PEEK_ONLINE_SCHEDULER": "1",
        "PEEK_ONLINE_CLPM": "1",
        "PEEK_ONLINE_CLPM_GROUP_MAJOR": "1",
        "PEEK_ONLINE_CLPM_DYNAMIC_LANE": "1",
        "PEEK_ONLINE_EVICTION": "1",
        "PEEK_ONLINE_EVICTION_MODE": "cluster",
    },
    "peek-offline": {
        "PEEK_OFFLINE_ENABLE": "1",
        "PEEK_OFFLINE_SERVER_REORDER": "1",
        "PEEK_QUEUE_AWARE": "1",
    },
}


def apply() -> None:
    name = os.environ.get("PEEK_PRESET", "").strip().lower()
    if not name:
        return
    bundle = _PRESETS.get(name)
    if bundle is None:
        warnings.warn(
            f"peek: PEEK_PRESET={name!r} is not a recognized preset; "
            f"valid values: {sorted(_PRESETS)}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    for key, value in bundle.items():
        os.environ.setdefault(key, value)
