import random


class ConfigCrossoverBatchSampler:
    def __init__(
        self,
        groups: list[list[int]],
        batch_size: int,
        shuffle_groups: bool = True,
        shuffle_within_groups: bool = True,
    ):
        # TODO: implement crossover. it's my idea, theres no link
        self.groups = groups
        self.batch_size = batch_size
        self.shuffle = shuffle_groups
        self.shuffle_within_groups = shuffle_within_groups

        self.batch_list: list[list[int]] = self.get_batches()

    def get_batches(self) -> list[list[int]]:
        """Returns a list of batches, where each batch is a list of indices."""
        if self.shuffle:
            random.shuffle(self.groups)

        self.batch_list: list[list[int]] = []

        for group in self.groups:
            if self.shuffle_within_groups:
                random.shuffle(group)

            for i in range(0, len(group), self.batch_size):
                self.batch_list.append(group[i : i + self.batch_size])

        random.shuffle(self.batch_list)
        return self.batch_list

    def __call__(self):
        for batch in self.batch_list:
            yield batch
