from aim_flow.eval_bench.generation import (
    _install_rectified_cfgpp_diffusers_compat,
    _validate_rectified_cfgpp_pipeline_import,
    apply_spfc_variant,
    load_bench_config,
    unload_model,
)


class FakePipe:
    def __init__(self):
        self.freed = False

    def maybe_free_model_hooks(self):
        self.freed = True


class FakeBackend:
    def __init__(self):
        self.pipe = FakePipe()


def test_bench_config_locks_sd3_medium_no_t5_seed_and_schedule():
    config = load_bench_config(seed=13)
    assert config.model.model_id == "stabilityai/stable-diffusion-3-medium-diffusers"
    assert config.model.load_t5_text_encoder is False
    assert config.sampler.seed == 13
    assert config.sampler.guidance_scale == 4.5
    assert config.primitive_flow.aggregation_steps == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 23]


def test_bench_config_accepts_explicit_guidance_scale():
    config = load_bench_config(seed=13, guidance_scale=1.0)
    assert config.sampler.guidance_scale == 1.0


def test_spfc_component_variants_mutate_only_expected_knobs():
    base = load_bench_config(seed=13)
    no_ltp = apply_spfc_variant(base, "no_ltp")
    no_source = apply_spfc_variant(base, "no_source_flow")
    steering = apply_spfc_variant(base, "steering_0.25")
    max_primitives = apply_spfc_variant(base, "max_primitives_2")

    assert no_ltp.primitive_flow.ltp_enabled is False
    assert no_ltp.primitive_flow.ltp_mode == "off"
    assert no_source.primitive_flow.include_source_flow is False
    assert steering.primitive_flow.steering_strength == 0.25
    assert max_primitives.primitive_flow.max_primitives == 2
    assert base.primitive_flow.ltp_enabled is True


def test_unload_model_calls_diffusers_offload_hook():
    backend = FakeBackend()
    pipe = backend.pipe
    unload_model(backend)
    assert pipe.freed


def test_rectified_cfgpp_diffusers_compat_installs_sd3_ip_adapter_mixin(monkeypatch):
    import diffusers.loaders as loaders

    monkeypatch.delattr(loaders, "SD3IPAdapterMixin", raising=False)
    _install_rectified_cfgpp_diffusers_compat()

    from diffusers.loaders import SD3IPAdapterMixin

    compat = SD3IPAdapterMixin()
    assert compat.is_ip_adapter_active is False


def test_rectified_cfgpp_pipeline_import_uses_compat_mixin(tmp_path, monkeypatch):
    import diffusers.loaders as loaders

    monkeypatch.delattr(loaders, "SD3IPAdapterMixin", raising=False)
    pipeline_dir = tmp_path / "rect-cfg-SD3-pipeline"
    pipeline_dir.mkdir()
    (pipeline_dir / "pipeline.py").write_text(
        "from diffusers.loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin\n"
        "class DummyRectifiedPipeline(SD3IPAdapterMixin):\n"
        "    pass\n",
        encoding="utf-8",
    )

    _install_rectified_cfgpp_diffusers_compat()
    _validate_rectified_cfgpp_pipeline_import(tmp_path)
