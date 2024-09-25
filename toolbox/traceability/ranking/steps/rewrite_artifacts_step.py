from toolbox.constants.default_model_managers import get_best_default_llm_manager_long_context
from toolbox.pipeline.abstract_pipeline_step import AbstractPipelineStep
from toolbox.summarize.artifact.artifact_summary_types import ArtifactSummaryTypes
from toolbox.summarize.artifact.artifacts_summarizer import ArtifactsSummarizer

from toolbox.traceability.ranking.common.ranking_args import RankingArgs
from toolbox.traceability.ranking.common.ranking_state import RankingState


class RewriteArtifactsStep(AbstractPipelineStep[RankingArgs, RankingState]):

    def _run(self, args: RankingArgs, state: RankingState) -> None:
        """
        Rewrites the artifacts to be in the same format before tracing.
        :param args: The ranking arguments to the pipeline.
        :param state: The state of the current pipeline.
        :return: NOne
        """
        if args.rewrite_artifacts:
            summarizer = ArtifactsSummarizer(summarize_code_only=False, code_summary_type=ArtifactSummaryTypes.NL_BASE,
                                             llm_manager_for_artifact_summaries=get_best_default_llm_manager_long_context())
            args.dataset.artifact_df.summarize_content(summarizer, summarize_from_existing=True)
