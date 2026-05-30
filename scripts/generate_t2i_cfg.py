"""Generate the local GenEval CFG comparison outputs."""

from t2i_generation_common import generate_method


if __name__ == "__main__":
    generate_method(method="cfg", title="SD3 CFG", default_guidance_scale=4.5, archive_filename="geneval_cfg.zip")
