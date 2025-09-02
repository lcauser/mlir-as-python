from mlir.ir.value import Value, OpResult, BlockArgument
from mlir.ir.types import TypeBase


class DummyType(TypeBase):
    def validate_type(self, value):
        pass

    def __str__(self) -> str:
        return "dummy"

    def __hash__(self) -> int:
        return hash("dummy")


class TestValue:
    def test_init(self):
        type = DummyType()
        value = Value(type)
        assert value.type == type
        assert value.uses == []

    def test_add_use(self):
        type = DummyType()
        value = Value(type)
        value.add_use(5)
        assert 5 in value.uses

    def test_remove_use(self):
        type = DummyType()
        value = Value(type)
        value.add_use(5)
        value.remove_use(5)
        assert 5 not in value.uses


class TestOpResult:
    def test_init(self):
        type = DummyType()
        op_result = OpResult(type, None, 0)
        assert op_result.type == type
        assert op_result.owner is None
        assert op_result.index == 0
        assert op_result.uses == []

    def test_repr(self):
        type = DummyType()
        op_result = OpResult(type, None, 0)
        repr_str = repr(op_result)
        assert "OpResult" in repr_str
        assert "type=dummy" in repr_str
        assert "owner=None" in repr_str
        assert "index=0" in repr_str


class TestBlockArgument:
    def test_init(self):
        type = DummyType()
        block_arg = BlockArgument(type, None, 0)
        assert block_arg.type == type
        assert block_arg.owner is None
        assert block_arg.index == 0
        assert block_arg.uses == []

    def test_repr(self):
        type = DummyType()
        block_arg = BlockArgument(type, None, 0)
        repr_str = repr(block_arg)
        assert "BlockArgument" in repr_str
        assert "type=dummy" in repr_str
        assert "owner=None" in repr_str
        assert "index=0" in repr_str
