class DatasetDescriptor:
    def __init__(
        self,
        input_column: str,
        target_column: str,
        other_columns: list[str] = [],  # noqa: B006
    ):
        self.input_column = input_column
        self.target_column = target_column
        self.other_columns = other_columns

    def __str__(self) -> str:
        return (
            f"\tinput column : {self.input_column} \n"
            f"\ttarget column : {self.target_column}"
        )


class JSONDatasetDescriptor(DatasetDescriptor):
    def __init__(
        self,
        input_column: str,
        target_column: str,
        path: str,
    ):
        super().__init__(
            input_column=input_column,
            target_column=target_column,
        )
        self.path = path

    def __str__(self) -> str:
        return f'JSON file: "{self.path}"'


class ExcelDatasetDescriptor(DatasetDescriptor):
    def __init__(
        self,
        input_column: str,
        target_column: str,
        path: str,
        sheet_name: str,
    ):
        super().__init__(
            input_column=input_column,
            target_column=target_column,
        )
        self.path = path
        self.sheet_name = sheet_name

    def __str__(self) -> str:
        return (
            super().__str__()
            + f'\n\texcel file: "{self.path}" from sheet "{self.sheet_name}"'
        )


class Datasets:
    PUBLICATION_DATASET = ExcelDatasetDescriptor(
        "IRRADIATION",
        "DC_POWER",
        "assets/datasets/publishable_data.xlsx",
        "Sheet1",
    )
