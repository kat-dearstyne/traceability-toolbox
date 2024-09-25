from toolbox.data.dataframes.artifact_dataframe import ArtifactDataFrame
from toolbox.data.keys.structure_keys import ArtifactKeys
from toolbox.data.tdatasets.prompt_dataset import PromptDataset
from toolbox.pipeline.abstract_pipeline_step import AbstractPipelineStep
from toolbox.summarize.artifact.artifacts_summarizer import ArtifactsSummarizer
from toolbox.summarize.summarizer_args import SummarizerArgs
from toolbox.summarize.summarizer_state import SummarizerState
from toolbox.summarize.summarizer_util import SummarizerUtil


class StepResummarizeArtifacts(AbstractPipelineStep[SummarizerArgs, SummarizerState]):

    def _run(self, args: SummarizerArgs, state: SummarizerState) -> None:
        """
        Re-summarizes the artifacts with the project summary.
        :param args: Arguments to summarizer pipeline.
        :param state: Current state of the summarizer pipeline.
        :return: None
        """
        if args.do_resummarize_artifacts and not args.no_project_summary:
            orig_artifact_df = state.dataset.artifact_df
            artifact_df = ArtifactDataFrame({ArtifactKeys.ID: orig_artifact_df.index,
                                             ArtifactKeys.CONTENT: orig_artifact_df[ArtifactKeys.CONTENT],
                                             ArtifactKeys.LAYER_ID: orig_artifact_df[ArtifactKeys.LAYER_ID]})
            params = SummarizerUtil.get_params_for_artifact_summarizer(args)
            summarizer = ArtifactsSummarizer(**params, project_summary=state.final_project_summary, summarizer_id="Second Summary")
            artifact_df.summarize_content(summarizer, re_summarize=True)
            state.re_summarized_artifacts_dataset = PromptDataset(artifact_df=artifact_df)
