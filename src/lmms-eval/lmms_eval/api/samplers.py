class ContextSampler:
    def __init__(self, docs, task, fewshot_indices=None, rnd=None) -> None:
        self.rnd = rnd
        assert self.rnd, "must pass rnd to FewShotSampler!"

        self.task = task
        self.config = task._config

        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        self.doc_to_text = self.task.doc_to_text
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.docs = docs  # HF dataset split, provided by task._fewshot_docs()
        if fewshot_indices:  # subset few-shot docs from
            self.docs = self.docs.select(fewshot_indices)

    def _select_fewshot_docs(self, doc, num_fewshot):
        """Sample fewshot docs, excluding the current doc if from same split."""
        n_samples = (
            num_fewshot + 1
            if self.config.fewshot_split == self.config.test_split
            else num_fewshot
        )
        fewshotex = self.sample(n_samples)
        return [x for x in fewshotex if x != doc][:num_fewshot]

    def _format_example_text(self, doc):
        """Format the text+target for a single fewshot example."""
        text_part = (
            self.doc_to_text(doc)
            if (
                self.config.doc_to_choice is None
                or type(self.doc_to_text(doc)) is str
            )
            else self.doc_to_choice(doc)[self.doc_to_text(doc)]
        )
        target = self.doc_to_target(doc)
        target_part = (
            str(target[0])
            if type(target) is list
            else (
                target
                if (
                    self.config.doc_to_choice is None
                    or type(target) is str
                )
                else str(self.doc_to_choice(doc)[target])
            )
        )
        return text_part + self.target_delimiter + target_part

    def get_context(self, doc, num_fewshot):
        selected_docs = self._select_fewshot_docs(doc, num_fewshot)
        labeled_examples = (
            self.fewshot_delimiter.join(
                [self._format_example_text(d) for d in selected_docs]
            )
            + self.fewshot_delimiter
        )
        return labeled_examples

    def get_multimodal_context(self, doc, num_fewshot):
        """Return fewshot context with both text and visuals.

        Each fewshot example's text is prefixed with ``<image>`` markers
        (one per image in that example) so models can align images to text.

        Returns:
            tuple: (labeled_examples_text: str, fewshot_visuals: list[list])
                - labeled_examples_text: text context with ``<image>`` markers
                - fewshot_visuals: list of visual lists, one per fewshot example.
                  Each entry is the return value of doc_to_visual for that example.
        """
        selected_docs = self._select_fewshot_docs(doc, num_fewshot)

        doc_to_visual = self.task.doc_to_visual

        # Build text + collect visuals per fewshot example
        example_texts = []
        fewshot_visuals = []
        for d in selected_docs:
            try:
                visuals = doc_to_visual(d)
            except Exception:
                visuals = []
            visuals = visuals if visuals else []
            fewshot_visuals.append(visuals)

            # Prefix text with <image> markers (one per visual)
            n_images = len(visuals)
            image_prefix = "<image> " * n_images if n_images > 0 else ""
            example_texts.append(
                image_prefix + self._format_example_text(d)
            )

        labeled_examples = (
            self.fewshot_delimiter.join(example_texts)
            + self.fewshot_delimiter
        )

        return labeled_examples, fewshot_visuals

    def sample(self, n):
        """
        Draw `n` samples from our fewshot docs. This method should be overridden by subclasses.
        """

        return self.rnd.sample(self.docs, n)


class FirstNSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(self.docs), f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        return self.docs[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        pass


class ManualSampler(ContextSampler):
    def sample(self, n) -> None:
        """ """
        pass


SAMPLER_REGISTRY = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
}


def get_sampler(name):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}")
