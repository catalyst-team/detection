from catalyst.dl import SupervisedRunner


class Runner(SupervisedRunner):
    def __init__(
        self,
        model=None,
        device=None,
        input_key=None,
        output_key=None
    ):
        super().__init__(
            model=model,
            device=device,
            input_key=input_key,
            output_key=output_key
        )
