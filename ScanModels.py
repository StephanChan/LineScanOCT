from dataclasses import asdict, dataclass


@dataclass
class SampleCenter:
    sample_id: int
    x: float
    y: float
    z: float

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(
            sample_id=int(data["sample_id"]),
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data["z"]),
        )


@dataclass
class FOVLocation:
    sample_id: int
    x: float
    y: float
    z: float = 0.0
    y_length_mm: float | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(
            sample_id=int(data["sample_id"]),
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data.get("z", 0.0)),
            y_length_mm=(
                None if data.get("y_length_mm") is None else float(data.get("y_length_mm"))
            ),
        )


def sample_center_records(items):
    return [
        item.to_dict() if isinstance(item, SampleCenter) else SampleCenter.from_dict(item).to_dict()
        for item in items
    ]


def fov_location_records(items):
    return [
        item.to_dict() if isinstance(item, FOVLocation) else FOVLocation.from_dict(item).to_dict()
        for item in items
    ]


def load_sample_centers(items):
    return [item if isinstance(item, SampleCenter) else SampleCenter.from_dict(item) for item in items]


def load_fov_locations(items):
    return [item if isinstance(item, FOVLocation) else FOVLocation.from_dict(item) for item in items]
