# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from granite_core.citations.types import Citation as GraniteCitation
from granite_core.usage import UsageInfo as GraniteUsageInfo

from http_agents.models.responses import Citation, UsageInfo


def convert_citation(granite_citation: GraniteCitation) -> Citation:
    """Convert granite_core Citation to API Citation model."""
    return Citation(
        url=granite_citation.url or "",
        title=granite_citation.title,
        description=granite_citation.context_text,
        start_index=granite_citation.start_index,
        end_index=granite_citation.end_index,
    )


def convert_usage_info(granite_usage: GraniteUsageInfo) -> UsageInfo:
    """Convert granite_core UsageInfo to API UsageInfo model."""
    return UsageInfo(
        prompt_tokens=granite_usage.prompt_tokens or 0,
        completion_tokens=granite_usage.completion_tokens or 0,
        total_tokens=granite_usage.total_tokens or 0,
        model_id=granite_usage.model_id,
    )
