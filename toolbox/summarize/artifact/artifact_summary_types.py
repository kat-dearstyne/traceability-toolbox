from enum import Enum

from toolbox.summarize.prompts.artifact_summary_prompts import CODE_SUMMARY, NL_SUMMARY


class ArtifactSummaryTypes(Enum):
    CODE_BASE = CODE_SUMMARY
    NL_BASE = NL_SUMMARY
