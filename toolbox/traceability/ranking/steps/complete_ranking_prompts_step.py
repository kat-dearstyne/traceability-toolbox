from typing import Dict

from toolbox.constants.ranking_constants import RANKING_PARENT_TAG
from toolbox.constants.summary_constants import ARTIFACT_HEADER
from toolbox.constants.symbol_constants import NEW_LINE
from toolbox.data.keys.prompt_keys import PromptKeys
from toolbox.data.keys.structure_keys import ArtifactKeys, TraceKeys
from toolbox.llm.anthropic_manager import AnthropicManager
from toolbox.llm.llm_responses import GenerationResponse
from toolbox.llm.llm_trainer import LLMTrainer
from toolbox.llm.prompts.multi_artifact_prompt import MultiArtifactPrompt
from toolbox.llm.prompts.prompt import Prompt
from toolbox.llm.prompts.prompt_builder import PromptBuilder
from toolbox.pipeline.abstract_pipeline_step import AbstractPipelineStep
from toolbox.traceability.ranking.common.ranking_args import RankingArgs
from toolbox.traceability.ranking.common.ranking_state import RankingState
from toolbox.traceability.ranking.prompts import QUESTION1, QUESTION2, RANKING_GOAL
from toolbox.util.enum_util import EnumDict
from toolbox.util.llm_response_util import LLMResponseUtil
from toolbox.util.prompt_util import PromptUtil
from toolbox.util.ranking_util import RankingUtil


class CompleteRankingPromptsStep(AbstractPipelineStep[RankingArgs, RankingState]):

    def _run(self, args: RankingArgs, state: RankingState) -> None:
        """
        Completes the ranking prompts.
        :param args: The pipeline arguments / configuration.
        :param state: The state of the current run.
        :return: None
        """
        generation_response = self.complete_ranking_prompts(args, state)
        state.ranking_responses = generation_response

    @staticmethod
    def complete_ranking_prompts(args: RankingArgs, state: RankingState) -> GenerationResponse:
        """
        Completes the ranking prompts.
        :param args: The pipeline configuration.
        :param state: The ranking store.
        :return: None
        """
        prompt_builder = CompleteRankingPromptsStep.create_ranking_prompt_builder(state)
        prompts = [CompleteRankingPromptsStep.create_prompts(p_name, state.artifact_map, prompt_builder, args, state)
                   for p_name in args.parent_ids]
        save_and_load_path = LLMResponseUtil.generate_response_save_and_load_path(
            state.get_path_to_state_checkpoint(args.export_dir), "ranking_response") if args.export_dir else args.export_dir
        predictions = LLMTrainer.predict_from_prompts(llm_manager=args.ranking_llm_model_manager, prompt_builders=prompt_builder,
                                                      message_prompts=prompts, save_and_load_path=save_and_load_path).predictions
        task_prompt = prompt_builder.prompts[-1]
        tag_for_response = task_prompt.response_manager.get_all_tag_ids()[0] if len(
            task_prompt.response_manager.get_all_tag_ids()) > 0 else None
        parsed_answers = LLMResponseUtil.extract_predictions_from_response(predictions,
                                                                           response_prompt_ids=task_prompt.args.prompt_id,
                                                                           tags_for_response=tag_for_response)

        return parsed_answers

    @staticmethod
    def create_prompts(parent_id: str, artifact_map: Dict, prompt_builder: PromptBuilder, args: RankingArgs,
                       state: RankingState) -> Prompt:
        """
        Creates ranking prompt for parent artifact.
        :param parent_id: The id of the parent to create prompt for.
        :param prompt_builder: The prompt builder to use to create the prompt
        :param artifact_map: Maps artifact id to content
        :param args: The arguments to the ranking pipeline
        :param state: The state of the current ranking run.
        :return: The ranking prompt.
        """
        max_children = args.max_children_per_query
        entries = state.get_current_parent2children()[parent_id][:max_children]
        parent_body = artifact_map[parent_id]
        artifacts = [EnumDict({ArtifactKeys.ID: i, ArtifactKeys.CONTENT: artifact_map[entry[TraceKeys.child_label()]]})
                     for i, entry in enumerate(entries)]

        prompt_dict = prompt_builder.build(model_format_args=AnthropicManager.prompt_args,
                                           parent_body=parent_body,
                                           artifacts=artifacts)
        prompt = prompt_dict[PromptKeys.PROMPT]

        return prompt

    @staticmethod
    def create_ranking_prompt_builder(state: RankingState) -> PromptBuilder:
        """
        Creates prompt builder for ranking artifacts.
        :param state: The state of the ranking pipeline.
        :return: The prompt builder used to rank candidate children artifacts.
        """
        prompt_builder = PromptBuilder(prompts=[
            RANKING_GOAL,
            Prompt(PromptUtil.create_xml(RANKING_PARENT_TAG, '{parent_body}', prefix=NEW_LINE, suffix=NEW_LINE)),
        ])

        RankingUtil.add_project_summary_prompt(prompt_builder, state)

        prompt_builder.add_prompt(MultiArtifactPrompt(prompt_start=PromptUtil.as_markdown_header(ARTIFACT_HEADER),
                                                      build_method=MultiArtifactPrompt.BuildMethod.XML,
                                                      include_ids=True))

        for q in (QUESTION1, QUESTION2):
            prompt_builder.add_prompt(q)

        return prompt_builder
