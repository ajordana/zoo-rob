from tasks.humanoid_balance_unconstrained import HumanoidBalanceUnconstrained


class HumanoidWalkUnconstrained(HumanoidBalanceUnconstrained):
    """Unitree G1 humanoid tracking a forward-walking mocap reference.

    Wraps HumanoidBalanceUnconstrained with a fixed clean-walk reference so the
    figures land in their own folder, separate from the balance task.
    """

    def __init__(
        self,
        reference_filename: str = "DefaultDatasets/mocap/UnitreeG1/walk.npz",
        start: int = 0,
    ) -> None:
        super().__init__(reference_filename=reference_filename, start=start)
