from fpv_vggt_lab.cloud_job import _runner_script


def test_cloud_runner_converts_cuda_tensors_before_point_cloud_unprojection():
    script = _runner_script()

    assert "def to_numpy_array(value):" in script
    assert 'to_numpy_array(predictions["depth"].squeeze(0))' in script
    assert "to_numpy_array(extrinsic.squeeze(0))" in script
    assert "to_numpy_array(intrinsic.squeeze(0))" in script


def test_cloud_runner_recursively_converts_nested_tensor_predictions():
    script = _runner_script()

    assert "if isinstance(value, dict):" in script
    assert "if isinstance(value, (list, tuple)):" in script
    assert "value = np.asarray(value)" in script
    assert "if value.dtype == object:" in script
    assert 'raise RuntimeError(f"could not convert prediction {key!r} to numpy: {exc}")' in script
