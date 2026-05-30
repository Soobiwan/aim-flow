"""Generate the local GenEval base SD3 comparison outputs."""

from t2i_generation_common import generate_method


if __name__ == "__main__":
    generate_method(method="base", title="Base SD3", default_guidance_scale=1.0, archive_filename="geneval_base.zip")
