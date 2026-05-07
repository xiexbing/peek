# Copyright 2026 Anonymous Authors
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

from dataclasses import dataclass, field
import time


@dataclass
class PromptRequest:
    id: str
    token_ids: list[int]
    arrival_time: float = field(default_factory=time.time)
    wait_count: int = 0
    metadata: dict = field(default_factory=dict)

    def age(self) -> float:
        """Return time in seconds since arrival."""
        return time.time() - self.arrival_time

    def increment_wait(self) -> None:
        """Bump wait count when skipped in a batch."""
        self.wait_count += 1
